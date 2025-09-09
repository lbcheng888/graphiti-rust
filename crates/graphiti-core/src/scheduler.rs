//! Minimal DAG scheduler and sub-agent concurrency governance skeleton
//! Minimal DAG scheduler and sub-agent concurrency governance skeleton
//!
//! Features:
//! - Ready-queue bridging via mpsc (consumer → feeder)
//! - Global concurrency limit + AgentPolicy (per-agent concurrency/budget)
//! - Budget consume (feeder) and optional refund on failure (consumer)

use std::collections::HashMap;
use std::collections::HashSet;

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

/// A unit of work executed by the scheduler
pub trait Task: Send + Sync + 'static {
    /// A stable identifier for deduplication
    fn id(&self) -> String;
    /// Estimated cost in tokens or milliseconds for budgeting
    fn estimated_cost(&self) -> u64 {
        1
    }
    /// Execute the task's work
    fn run(
        &self,
        cancel: CancellationToken,
    ) -> Box<dyn Future<Output = anyhow::Result<serde_json::Value>> + Send + Unpin>;
}

/// Edge in DAG: parent must complete before child starts
/// Edge in the DAG where `parent` must complete before `child` can start
#[derive(Debug, Clone)]
pub struct Dependency {
    /// The prerequisite task ID
    pub parent: String,
    /// The dependent task ID
    pub child: String,
}

/// Scheduler configuration
/// Configuration for the scheduler runtime
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of tasks allowed to run concurrently
    pub max_concurrency: usize,
    /// Per-task timeout
    pub task_timeout: Duration,
    /// Bounded channel capacity for results/messages
    pub queue_bound: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            task_timeout: Duration::from_secs(30),
            queue_bound: 1024,
        }
    }
}

/// Governance policy for sub-agents (budget/priority)
/// Governance policy controlling concurrency and budget for sub-agents
#[derive(Debug, Clone, Default)]
pub struct AgentPolicy {
    /// Max number of parallel tasks a sub-agent may run
    pub max_parallel_tasks: usize,
    /// Budget available to the sub-agent (tokens/ms units)
    pub max_budget: u64,
    /// If true, refund budget on failure/cancel
    pub refund_on_fail: bool,

    /// Cost per task (ms or token units). If set, budget consumption sums estimated_cost
    pub cost_per_task: Option<u64>,
}

/// Dag specification
/// DAG specification consumed by the scheduler
#[derive(Clone)]
pub struct DagSpec {
    /// All tasks to schedule
    pub tasks: Vec<Arc<dyn Task>>,
    /// Dependencies between tasks
    pub deps: Vec<Dependency>,
}

/// Scheduler handle
pub struct Scheduler {
    policy: AgentPolicy,
    budget_left: Arc<tokio::sync::Mutex<u64>>,

    cfg: SchedulerConfig,
    concurrency: Arc<Semaphore>,
}

impl Scheduler {
    #[must_use]
    /// Create a new scheduler with specified configuration
    pub fn new_with_policy(cfg: SchedulerConfig, policy: AgentPolicy) -> Self {
        let concurrency = Arc::new(Semaphore::new(cfg.max_concurrency));
        let budget_left = Arc::new(tokio::sync::Mutex::new(policy.max_budget));
        Self {
            cfg,
            concurrency,
            policy,
            budget_left,
        }
    }

    /// Create a new scheduler with default agent policy
    ///
    /// Uses `max_concurrency` for the agent parallelism, unlimited budget, and refunds
    /// budget on failure. This is a convenience for typical scenarios.
    pub fn new(cfg: SchedulerConfig) -> Self {
        let c = cfg.clone();
        Self::new_with_policy(
            cfg,
            AgentPolicy {
                max_parallel_tasks: c.max_concurrency,
                max_budget: u64::MAX,
                refund_on_fail: true,
                cost_per_task: None,
            },
        )
    }

    /// Execute a DAG respecting dependencies; returns per-task results map
    ///
    /// # Panics
    /// If a dependency references a task id which is not present in `dag.tasks`.
    pub async fn execute(
        &self,
        dag: DagSpec,
        cancel: CancellationToken,
    ) -> HashMap<String, anyhow::Result<serde_json::Value>> {
        let mut indeg: HashMap<String, usize> = HashMap::new();
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        let mut task_by_id: HashMap<String, Arc<dyn Task>> = HashMap::new();
        for t in &dag.tasks {
            indeg.insert(t.id(), 0);
            task_by_id.insert(t.id(), t.clone());
        }
        for d in &dag.deps {
            *indeg.entry(d.child.clone()).or_insert(0) += 1;
            children
                .entry(d.parent.clone())
                .or_default()
                .push(d.child.clone());
        }

        let _total_tasks = dag.tasks.len();

        // 就绪队列桥接：使用有界通道连接 consumer → feeder
        let (ready_tx, ready_rx) = mpsc::channel::<String>(self.cfg.queue_bound);
        // 发送初始就绪任务
        for (k, &v) in &indeg {
            if v == 0 {
                match ready_tx.try_send(k.clone()) {
                    Ok(_) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        let txc = ready_tx.clone();
                        let kid = k.clone();
                        tokio::spawn(async move {
                            let _ = tokio::time::timeout(Duration::from_millis(200), txc.send(kid))
                                .await;
                        });
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!("ready queue closed while seeding");
                    }
                }
            }
        }
        // 将发送者所有权只保留在 consumer 侧，避免 feeder 持续等待
        let ready_tx_consumer = ready_tx.clone();
        drop(ready_tx);

        // 并发执法：同时运行数受限于 policy.max_parallel_tasks
        let agent_sem = Arc::new(Semaphore::new(self.policy.max_parallel_tasks));

        let (tx, mut rx) =
            mpsc::channel::<(String, anyhow::Result<serde_json::Value>)>(self.cfg.queue_bound);
        let results = Arc::new(tokio::sync::Mutex::new(HashMap::<
            String,
            anyhow::Result<serde_json::Value>,
        >::new()));
        let mut in_flight: HashSet<String> = HashSet::new();

        // feeder loop（从 ready_rx 拉取新就绪任务）

        let concurrency = self.concurrency.clone();
        let task_by_id_arc = Arc::new(task_by_id);
        let timeout = self.cfg.task_timeout;
        let feeder_cancel = cancel.clone();
        let mut ready_rx_feeder = ready_rx;
        let feeder = async move {
            while let Some(task_id) = ready_rx_feeder.recv().await {
                if feeder_cancel.is_cancelled() {
                    break;
                }
                if in_flight.contains(&task_id) {
                    continue;
                }

                // Agent 并发许可
                let Ok(agent_p) = agent_sem.clone().acquire_owned().await else {
                    break;
                };
                // 总并发许可
                let Ok(permit) = concurrency.clone().acquire_owned().await else {
                    break;
                };

                in_flight.insert(task_id.clone());
                let tx2 = tx.clone();
                let cancel_child = feeder_cancel.child_token();
                let t = task_by_id_arc.get(&task_id).cloned().unwrap();
                let budget_left = self.budget_left.clone();
                let policy = self.policy.clone();

                tokio::spawn(async move {
                    // 预算执法：消耗估算成本
                    if let Some(cost_per_task) = policy.cost_per_task {
                        let mut b = budget_left.lock().await;
                        if *b < cost_per_task {
                            // 预算不足：返回错误结果，便于 consumer 收敛和计数
                            let _ = tx2
                                .send((t.id(), Err(anyhow::anyhow!("budget exhausted"))))
                                .await;
                            drop(agent_p);
                            drop(permit);
                            return Ok(());
                        }
                        *b -= cost_per_task;
                    }

                    let fut = t.run(cancel_child.clone());
                    let res = tokio::time::timeout(timeout, fut)
                        .await
                        .map_err(|_| anyhow::anyhow!("task timed out"))?;
                    let _ = tx2.send((t.id(), res)).await;
                    drop(permit);
                    drop(agent_p);
                    Ok::<(), anyhow::Error>(())
                });
            }
            drop(tx);
            Ok::<(), anyhow::Error>(())
        };

        // consumer loop（处理完成事件并回送新就绪任务）
        let mut indeg_mut = indeg;
        let mut children_mut = children;
        let results_arc = results.clone();
        let consumer_cancel = cancel.clone();
        let consumer = async move {
            while let Some((tid, res)) = rx.recv().await {
                // 预算回补：失败/取消且配置 refund_on_fail 时回补
                // 注：此处仅在 cost_per_task 模式下回补固定成本；更复杂策略可基于 res 内元数据
                if self.policy.refund_on_fail {
                    if let Some(cost) = self.policy.cost_per_task {
                        if res.is_err() {
                            let mut b = self.budget_left.lock().await;
                            *b = b.saturating_add(cost);
                        }
                        // 系统级异常可在此记录/外传（例如通过一个全局非阻塞通道/metrics），避免静默
                    }
                }

                results_arc.lock().await.insert(tid.clone(), res);
                if let Some(ch) = children_mut.remove(&tid) {
                    for c in ch {
                        if let Some(v) = indeg_mut.get_mut(&c) {
                            *v = v.saturating_sub(1);
                            if *v == 0 {
                                let _ = ready_tx_consumer.send(c).await;
                            }
                        }
                    }
                }
                if consumer_cancel.is_cancelled() {
                    break;
                }
            }
            Ok::<(), anyhow::Error>(())
        };

        let _ = tokio::join!(feeder, consumer);
        // running was moved into feeder; no tasks remain here to shut down
        let guard = results.lock().await;
        guard
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_ref()
                        .cloned()
                        .map_err(|e| anyhow::anyhow!(e.to_string())),
                )
            })
            .collect()
    }
}

/// Example concrete task from a closure
/// Example concrete task from a closure
///
/// A convenience wrapper implementing `Task` for an async closure to simplify tests and examples.
pub struct FnTask<F>
where
    F: Fn(CancellationToken) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = anyhow::Result<serde_json::Value>> + Send + 'static,
{
    id: String,
    f: Arc<F>,
}

/// Boxed future type used by `FnTask`
pub type Fut = Box<dyn Future<Output = anyhow::Result<serde_json::Value>> + Send + Unpin>;

impl<F> FnTask<F>
where
    F: Fn(CancellationToken) -> Fut + Send + Sync + 'static,
{
    /// Create a `FnTask` from an identifier and an async closure
    pub fn new(id: impl Into<String>, f: F) -> Self {
        Self {
            id: id.into(),
            f: Arc::new(f),
        }
    }
}

impl<F> Task for FnTask<F>
where
    F: Fn(CancellationToken) -> Fut + Send + Sync + 'static,
{
    fn id(&self) -> String {
        self.id.clone()
    }
    fn run(
        &self,
        cancel: CancellationToken,
    ) -> Box<dyn Future<Output = anyhow::Result<serde_json::Value>> + Send + Unpin> {
        (self.f)(cancel)
    }
}
