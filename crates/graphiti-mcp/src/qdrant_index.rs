use anyhow::Result;
use qdrant_client::qdrant::{Distance, PointStruct, Value};
use qdrant_client::qdrant::{CreateCollectionBuilder, VectorParamsBuilder, UpsertPointsBuilder, SearchParams};
use qdrant_client::qdrant::{QueryPointsBuilder, Query};
use qdrant_client::qdrant::{DeletePointsBuilder, Filter, Condition};
use qdrant_client::qdrant::value::Kind as Vk;
use qdrant_client::Qdrant;
use serde::Deserialize;
use tokio::runtime::Handle;

#[derive(Clone)]
pub struct QdrantIndex {
    client: qdrant_client::Qdrant,
    rest_base: String,
}

#[derive(Debug, Clone)]
pub struct VectorInsert {
    pub id: String,
    pub vector: Vec<f32>,
    pub content: String,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
    pub file_extension: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VectorHit {
    pub score: f32,
    pub content: String,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
}

impl QdrantIndex {
    pub fn new(url: &str) -> Result<Self> {
        let client = Qdrant::from_url(url).build()?;
        let rest_base = if let Some(pos) = url.rfind(':') {
            let (host, port) = url.split_at(pos);
            if port.contains("6334") { format!("{}:6333", host) } else { format!("{}", url) }
        } else {
            url.to_string()
        };
        Ok(Self { client, rest_base })
    }

    pub fn ensure_collection(&self, name: &str, dimension: usize) -> Result<()> {
        run_async_blocking(async {
            match self.client.collection_exists(name).await {
                Ok(exists) => {
                    if !exists {
                        let req = CreateCollectionBuilder::new(name)
                            .vectors_config(VectorParamsBuilder::new(dimension as u64, Distance::Cosine));
                        if let Err(e) = self.client.create_collection(req).await {
                            eprintln!("[warn] gRPC create_collection failed: {} — trying REST", e);
                            let url = format!("{}/collections/{}", self.rest_base.trim_end_matches('/'), name);
                            let body = serde_json::json!({
                                "vectors": { "size": dimension as u64, "distance": "Cosine" }
                            });
                            let client = reqwest::Client::new();
                            let resp = client.put(url).json(&body).send().await?;
                            if !resp.status().is_success() {
                                anyhow::bail!("REST create_collection HTTP {}", resp.status());
                            }
                        }
                    }
                    Ok(())
                }
                Err(e) => {
                    eprintln!("[warn] gRPC collection_exists failed: {} — checking via REST", e);
                    let url = format!("{}/collections/{}", self.rest_base.trim_end_matches('/'), name);
                    let client = reqwest::Client::new();
                    let resp = client.get(&url).send().await?;
                    if resp.status() == reqwest::StatusCode::NOT_FOUND {
                        let body = serde_json::json!({
                            "vectors": { "size": dimension as u64, "distance": "Cosine" }
                        });
                        let resp2 = client.put(url).json(&body).send().await?;
                        if !resp2.status().is_success() {
                            anyhow::bail!("REST create_collection HTTP {}", resp2.status());
                        }
                    }
                    Ok(())
                }
            }
        })
    }

    pub fn insert(&self, name: &str, points: &[VectorInsert]) -> Result<()> {
        if points.is_empty() { return Ok(()); }
        let pts: Vec<PointStruct> = points
            .iter()
            .map(|p| {
                let mut hasher = md5::Md5::new();
                use md5::Digest as _;
                hasher.update(p.id.as_bytes());
                let digest = hasher.finalize();
                let id_num = u64::from_be_bytes([digest[0],digest[1],digest[2],digest[3],digest[4],digest[5],digest[6],digest[7]]);
                let mut payload: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
                payload.insert("content".to_string(), Value { kind: Some(Vk::StringValue(p.content.clone())) });
                payload.insert("relative_path".to_string(), Value { kind: Some(Vk::StringValue(p.relative_path.clone())) });
                payload.insert("start_line".to_string(), Value { kind: Some(Vk::IntegerValue(p.start_line as i64)) });
                payload.insert("end_line".to_string(), Value { kind: Some(Vk::IntegerValue(p.end_line as i64)) });
                payload.insert("language".to_string(), Value { kind: Some(Vk::StringValue(p.language.clone())) });
                payload.insert("file_extension".to_string(), Value { kind: Some(Vk::StringValue(p.file_extension.clone())) });
                PointStruct::new(id_num, p.vector.clone(), payload)
            })
            .collect();

        run_async_blocking(async move {
            let req = UpsertPointsBuilder::new(name, pts.clone()).wait(true);
            match self.client.upsert_points(req).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    eprintln!("[warn] gRPC upsert failed: {} — falling back to REST /points/upsert", e);
                    let mut rest_points = Vec::with_capacity(points.len());
                    for p in points.iter() {
                        let mut hasher = md5::Md5::new();
                        use md5::Digest as _;
                        hasher.update(p.id.as_bytes());
                        let digest = hasher.finalize();
                        let id_num = u64::from_be_bytes([digest[0],digest[1],digest[2],digest[3],digest[4],digest[5],digest[6],digest[7]]);
                        let payload = serde_json::json!({
                            "content": p.content,
                            "relative_path": p.relative_path,
                            "start_line": p.start_line as i64,
                            "end_line": p.end_line as i64,
                            "language": p.language,
                            "file_extension": p.file_extension,
                        });
                        rest_points.push(serde_json::json!({
                            "id": id_num,
                            "vector": p.vector,
                            "payload": payload,
                        }));
                    }
                    let body = serde_json::json!({ "points": rest_points, "wait": true });
                    let url = format!("{}/collections/{}/points/upsert", self.rest_base.trim_end_matches('/'), name);
                    let client = reqwest::Client::new();
                    let resp = client.post(url).json(&body).send().await?;
                    if !resp.status().is_success() { anyhow::bail!("REST upsert HTTP {}", resp.status()); }
                    Ok(())
                }
            }
        })
    }

    pub fn search(&self, name: &str, vector: &[f32], top_k: usize) -> Result<Vec<VectorHit>> {
        run_async_blocking(async move {
            let sp = SearchParams { hnsw_ef: Some(128), exact: Some(true), quantization: None, indexed_only: Some(false) };
            let req = QueryPointsBuilder::new(name)
                .query(Query::new_nearest(vector.to_vec()))
                .limit(top_k as u64)
                .with_payload(true)
                .params(sp);
            let result = match self.client.query(req).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[warn] gRPC query failed: {} — falling back to REST", e);
                    return self.search_rest_inner(name, vector, top_k).await;
                }
            };
            let out = result.result.into_iter().filter_map(|p| {
                let payload = p.payload;
                let str_of = |k: &str| -> String {
                    payload.get(k).and_then(|v| match &v.kind { Some(Vk::StringValue(s)) => Some(s.clone()), _ => None }).unwrap_or_default()
                };
                let int_of = |k: &str| -> i64 { payload.get(k).and_then(|v| match v.kind { Some(Vk::IntegerValue(n)) => Some(n), _ => None }).unwrap_or(0) };
                Some(VectorHit {
                    score: p.score as f32,
                    content: str_of("content"),
                    relative_path: str_of("relative_path"),
                    start_line: int_of("start_line") as u32,
                    end_line: int_of("end_line") as u32,
                    language: str_of("language"),
                })
            }).collect();
            Ok(out)
        })
    }

    pub fn delete_by_relative_paths(&self, name: &str, rel_paths: &[String]) -> Result<()> {
        if rel_paths.is_empty() { return Ok(()); }
        run_async_blocking(async move {
            let conditions: Vec<qdrant_client::qdrant::Condition> = rel_paths.iter().map(|p| Condition::matches("relative_path", p.clone())).collect();
            let filter = if conditions.len() == 1 { Filter::must(conditions) } else { Filter::should(conditions) };
            let req = DeletePointsBuilder::new(name).points(filter.clone()).wait(true);
            match self.client.delete_points(req).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    eprintln!("[warn] gRPC delete failed: {} — falling back to REST", e);
                    let url = format!("{}/collections/{}/points/delete", self.rest_base.trim_end_matches('/'), name);
                    let conditions_json: Vec<serde_json::Value> = rel_paths.iter().map(|p| serde_json::json!({"key": "relative_path","match": {"value": p}})).collect();
                    let filter_json = if conditions_json.len() == 1 { serde_json::json!({"must": conditions_json}) } else { serde_json::json!({"should": conditions_json}) };
                    let body = serde_json::json!({ "filter": filter_json });
                    let client = reqwest::Client::new();
                    let resp = client.post(url).json(&body).send().await?;
                    if !resp.status().is_success() { anyhow::bail!("REST delete HTTP {}", resp.status()); }
                    Ok(())
                }
            }
        })
    }

    async fn search_rest_inner(&self, name: &str, vector: &[f32], top_k: usize) -> Result<Vec<VectorHit>> {
        #[derive(Deserialize)]
        struct ScoredPoint { score: f32, payload: std::collections::HashMap<String, serde_json::Value> }
        #[derive(Deserialize)]
        struct SearchResponse { result: Vec<ScoredPoint> }
        let url = format!("{}/collections/{}/points/search", self.rest_base.trim_end_matches('/'), name);
        let body = serde_json::json!({ "vector": vector, "limit": top_k as u64, "with_payload": true, "params": {"hnsw_ef": 128, "exact": true} });
        let client = reqwest::Client::new();
        let resp = client.post(url).json(&body).send().await?;
        if !resp.status().is_success() { anyhow::bail!("REST search HTTP {}", resp.status()); }
        let sr: SearchResponse = resp.json().await?;
        let out = sr.result.into_iter().map(|p| {
            let payload = p.payload;
            let str_of = |k: &str| -> String { payload.get(k).and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default() };
            let int_of = |k: &str| -> i64 { payload.get(k).and_then(|v| v.as_i64()).unwrap_or(0) };
            VectorHit { score: p.score, content: str_of("content"), relative_path: str_of("relative_path"), start_line: int_of("start_line") as u32, end_line: int_of("end_line") as u32, language: str_of("language") }
        }).collect();
        Ok(out)
    }
}

fn run_async_blocking<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    if Handle::try_current().is_ok() {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread().enable_io().enable_time().build()?;
            rt.block_on(fut)
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread().enable_io().enable_time().build()?;
        rt.block_on(fut)
    }
}

