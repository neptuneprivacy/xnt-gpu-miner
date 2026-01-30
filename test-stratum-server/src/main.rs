use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tracing::{error, info, warn};
use reqwest::Client;
use base64::{Engine as _, engine::general_purpose};
use num_bigint::BigUint;
use num_traits::One;
use serde_json::Value;

const DIGEST_BYTES: usize = 5 * 8;
const DIGEST_HEX_LEN: usize = DIGEST_BYTES * 2;

// Simplified schema structure
// Digest is represented as a hex string or array of u64
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Digest {
    String(String),
    Array(Vec<u64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowMastPaths {
    #[serde(default)]
    pub pow: Option<PowPaths>,
    #[serde(default)]
    pub header: Option<HeaderPaths>,
    #[serde(default)]
    pub kernel: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowPaths {
    #[serde(default)]
    pub kernel_body: Vec<String>,
    #[serde(default)]
    pub type_scripts: Vec<String>,
    #[serde(default)]
    pub kernel: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderPaths {
    #[serde(default)]
    pub body: Vec<String>,
    #[serde(default)]
    pub appendix: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Digest,
    pub paths: PowMastPaths,
    pub difficulty: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum Request {
    Login {
        name: String,
        address: String,
        password: Option<String>,
        agent: String,
    },
    Keepalived {},
    Submit {
        worker: usize,
        id: String,
        pow: Box<BlockPow>,
    },
    // Notifications (without id on JsonRequest)
    Job {
        #[serde(flatten)]
        job: Box<Job>,
    },
    Pause {},
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(flatten)]
    pub request: Request,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPow {
    pub nonce: Vec<u64>,
    pub root: Vec<u64>,
    pub authentication_path_a: Vec<Vec<u64>>,
    pub authentication_path_b: Vec<Vec<u64>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Response {
    Login {
        id: usize,
        job: Box<Option<Job>>,
    },
    Keepalived {},
    Submit {
        success: bool,
    },
}

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum Error {
    Parse = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    Internal = -32603,
    OtherUnknown = 20,
    HeaderNotFound = 21,
    DuplicateShare = 22,
    LowDifficultyShare = 23,
    UnauthorizedWorker = 24,
}

impl Error {
    pub fn message(&self) -> &'static str {
        match self {
            Error::Parse => "Parse error",
            Error::InvalidRequest => "Invalid request",
            Error::MethodNotFound => "Method not found",
            Error::InvalidParams => "Invalid params",
            Error::Internal => "Internal error",
            Error::OtherUnknown => "Unknown error",
            Error::HeaderNotFound => "Header not found",
            Error::DuplicateShare => "Duplicate share",
            Error::LowDifficultyShare => "Low difficulty share",
            Error::UnauthorizedWorker => "Unauthorized worker",
        }
    }
}

#[derive(Serialize)]
pub struct JsonError {
    code: i32,
    message: &'static str,
}

impl From<Error> for JsonError {
    fn from(err: Error) -> JsonError {
        JsonError {
            code: err as i32,
            message: err.message(),
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum JsonResult {
    Result { result: Response },
    Error { error: JsonError },
}

#[derive(Serialize)]
pub struct JsonResponse {
    pub id: Option<u64>,
    pub jsonrpc: &'static str,
    #[serde(flatten)]
    pub result: JsonResult,
}

impl JsonResponse {
    pub fn new(id: Option<u64>, result: Response) -> Self {
        Self {
            id,
            jsonrpc: "2.0",
            result: JsonResult::Result { result },
        }
    }

    pub fn from_err(id: Option<u64>, error: Error) -> Self {
        Self {
            id,
            jsonrpc: "2.0",
            result: JsonResult::Error {
                error: error.into(),
            },
        }
    }
}

struct ClientState {
    worker_id: usize,
    address: String,
    name: String,
    agent: String,
    connected: bool,
    writer: Option<tokio::sync::mpsc::UnboundedSender<String>>,
}

struct RpcConfig {
    url: String,
    user: String,
    password: String,
    guesser_address: String,
    fetch_interval_sec: u64,
}

struct ServerState {
    clients: Arc<Mutex<HashMap<usize, ClientState>>>,
    next_worker_id: Arc<Mutex<usize>>,
    current_job: Arc<Mutex<Option<Job>>>,
    submitted_shares: Arc<Mutex<Vec<String>>>,
    rpc_config: Option<RpcConfig>,
    http_client: Client,
}

impl ServerState {
    fn new(rpc_config: Option<RpcConfig>) -> Self {
        Self {
            clients: Arc::new(Mutex::new(HashMap::new())),
            next_worker_id: Arc::new(Mutex::new(1)),
            current_job: Arc::new(Mutex::new(None)),
            submitted_shares: Arc::new(Mutex::new(Vec::new())),
            rpc_config,
            http_client: Client::new(),
        }
    }

    fn create_test_job() -> Job {
        // Create a simple test job
        Job {
            id: Digest::String("test_job_id_1234567890abcdef".to_string()),
            paths: PowMastPaths {
                pow: Some(PowPaths {
                    kernel_body: vec!["path1".to_string(), "path2".to_string()],
                    type_scripts: vec!["script1".to_string()],
                    kernel: vec!["kernel1".to_string()],
                }),
                header: Some(HeaderPaths {
                    body: vec!["header_body1".to_string()],
                    appendix: vec!["appendix1".to_string()],
                }),
                kernel: vec!["kernel_path1".to_string()],
            },
            difficulty: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string(),
        }
    }

    async fn fetch_job_from_rpc(&self) -> anyhow::Result<Option<Job>> {
        let config = match &self.rpc_config {
            Some(c) => c,
            None => return Ok(None),
        };

        // Build JSON-RPC request
        let request = json!({
            "jsonrpc": "2.0",
            "method": "mining_getBlockTemplate",
            "params": [config.guesser_address],
            "id": 1
        });

        // Build auth header
        let auth = if !config.user.is_empty() && !config.password.is_empty() {
            let credentials = format!("{}:{}", config.user, config.password);
            let encoded = general_purpose::STANDARD.encode(credentials.as_bytes());
            Some(format!("Basic {}", encoded))
        } else {
            None
        };

        // Make HTTP request
        let mut req = self.http_client
            .post(&config.url)
            .header("Content-Type", "application/json")
            .json(&request);

        if let Some(auth_header) = auth {
            req = req.header("Authorization", auth_header);
        }

        let response = req.send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("RPC request failed: {}", response.status()));
        }

        let json_response: serde_json::Value = response.json().await?;

        // Parse response: { "jsonrpc": "2.0", "result": { "template": { "metadata": {...} } } }
        if let Some(result) = json_response.get("result") {
            if let Some(template) = result.get("template") {
                if !template.is_null() {
                    if let Some(metadata) = template.get("metadata") {
                        return Ok(Some(parse_rpc_metadata_to_job(metadata)?));
                    }
                }
            }
        }

        Ok(None)
    }
}

fn bytes_to_hex_le(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

fn hex_to_le_bytes(hex: &str) -> Option<Vec<u8>> {
    let mut clean = hex.trim_start_matches("0x").trim().to_string();
    if clean.len() % 2 != 0 {
        clean = format!("0{}", clean);
    }

    let mut bytes = Vec::with_capacity(clean.len() / 2);
    let mut i = 0;
    while i + 1 < clean.len() {
        let byte = u8::from_str_radix(&clean[i..i + 2], 16).ok()?;
        bytes.push(byte);
        i += 2;
    }

    if bytes.len() < DIGEST_BYTES {
        bytes.resize(DIGEST_BYTES, 0);
    } else if bytes.len() > DIGEST_BYTES {
        bytes.truncate(DIGEST_BYTES);
    }

    Some(bytes)
}

fn digest_array_to_hex(values: &[u64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        for byte in 0..8 {
            bytes.push(((value >> (byte * 8)) & 0xff) as u8);
        }
    }

    if bytes.len() < DIGEST_BYTES {
        bytes.resize(DIGEST_BYTES, 0);
    } else if bytes.len() > DIGEST_BYTES {
        bytes.truncate(DIGEST_BYTES);
    }

    bytes_to_hex_le(&bytes)
}

fn extract_threshold_hex(metadata: &Value) -> Option<String> {
    let threshold_value = metadata.get("threshold")?;
    if let Some(s) = threshold_value.as_str() {
        let bytes = hex_to_le_bytes(s)?;
        return Some(bytes_to_hex_le(&bytes));
    }

    if let Some(array) = threshold_value.as_array() {
        let mut values = Vec::with_capacity(array.len());
        for item in array {
            let number = item.as_u64()?;
            values.push(number);
        }
        return Some(digest_array_to_hex(&values));
    }

    None
}

fn parse_rpc_metadata_to_job(metadata: &serde_json::Value) -> anyhow::Result<Job> {
    // Extract digest (proposal ID)
    let id = metadata
        .get("digest")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing digest in metadata"))?;

    // Extract pow_mast_paths
    let pow_mast_paths = metadata
        .get("pow_mast_paths")
        .or_else(|| metadata.get("powMastPaths"))
        .ok_or_else(|| anyhow::anyhow!("Missing pow_mast_paths in metadata"))?;

    // Parse paths structure
    let mut paths = parse_pow_mast_paths(pow_mast_paths)?;
    
    // Ensure all required fields are present (even if empty)
    // This ensures the Job structure is valid even if some paths are missing
    if paths.pow.is_none() {
        paths.pow = Some(PowPaths {
            kernel_body: Vec::new(),
            type_scripts: Vec::new(),
            kernel: Vec::new(),
        });
    }
    if paths.header.is_none() {
        paths.header = Some(HeaderPaths {
            body: Vec::new(),
            appendix: Vec::new(),
        });
    }

    // Extract difficulty from RPC threshold
    // For pool mode, we need to send an EASIER difficulty (like 10000X easier) for share validation
    // The RPC threshold is for full block solutions, but pool needs easier shares
    let rpc_threshold = extract_threshold_hex(metadata).unwrap_or_else(|| {
        warn!("Missing or invalid threshold in metadata; using max threshold");
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string()
    });
    
    // Convert threshold to easier pool difficulty for share validation
    // Pool needs much easier difficulty (e.g., 10000X easier) so miners can find shares
    // For now, use a fixed easy threshold - in production, calculate from RPC threshold
    let pool_difficulty = make_pool_difficulty(&rpc_threshold, 10000.0);

    info!("Parsed job: id={}, rpc_threshold={} (len={}), pool_difficulty={} (len={}), pow_paths={}, header_paths={}, kernel_paths={}",
        id,
        &rpc_threshold[..std::cmp::min(16, rpc_threshold.len())],
        rpc_threshold.len(),
        &pool_difficulty[..std::cmp::min(16, pool_difficulty.len())],
        pool_difficulty.len(),
        paths.pow.as_ref().map(|p| p.kernel_body.len() + p.type_scripts.len() + p.kernel.len()).unwrap_or(0),
        paths.header.as_ref().map(|h| h.body.len() + h.appendix.len()).unwrap_or(0),
        paths.kernel.len()
    );

    Ok(Job {
        id: Digest::String(id.to_string()),
        paths,
        difficulty: pool_difficulty,
    })
}

// Make pool difficulty easier (multiply threshold by factor)
// For pool mining, we need much easier difficulty (e.g., 10000X easier) for share validation
// This allows miners to find shares more frequently than full block solutions
// In mining: higher threshold value = easier difficulty
// Formula: pool_threshold = min(rpc_threshold * factor, MAX_THRESHOLD)
fn make_pool_difficulty(rpc_threshold: &str, factor: f64) -> String {
    let threshold_bytes = match hex_to_le_bytes(rpc_threshold) {
        Some(bytes) => bytes,
        None => {
            warn!(
                "Failed to parse threshold '{}', using max difficulty",
                rpc_threshold
            );
            return "f".repeat(DIGEST_HEX_LEN);
        }
    };
    let threshold_value = BigUint::from_bytes_le(&threshold_bytes);

    let factor_u64 = if factor.is_finite() && factor >= 1.0 {
        factor.round() as u64
    } else {
        1
    };

    let max_threshold = (BigUint::one() << (DIGEST_BYTES * 8)) - BigUint::one();
    let multiplied_threshold = threshold_value * BigUint::from(factor_u64);
    let pool_threshold = if multiplied_threshold > max_threshold {
        max_threshold
    } else {
        multiplied_threshold
    };

    let mut bytes = pool_threshold.to_bytes_le();
    if bytes.len() < DIGEST_BYTES {
        bytes.resize(DIGEST_BYTES, 0);
    } else if bytes.len() > DIGEST_BYTES {
        bytes.truncate(DIGEST_BYTES);
    }

    bytes_to_hex_le(&bytes)
}

fn parse_pow_mast_paths(paths_json: &serde_json::Value) -> anyhow::Result<PowMastPaths> {
    let mut paths = PowMastPaths {
        pow: None,
        header: None,
        kernel: Vec::new(),
    };

    // RPC format: pow and header are arrays, but pool protocol expects objects
    // Check if this is RPC format (arrays) or pool protocol format (objects)
    
    // Parse pow section - handle both RPC format (array) and pool protocol format (object)
    if let Some(pow) = paths_json.get("pow") {
        if pow.is_object() {
            // Pool protocol format: { "kernel_body": [...], "type_scripts": [...], "kernel": [...] }
            paths.pow = Some(PowPaths {
                kernel_body: parse_string_array(pow.get("kernel_body").or_else(|| pow.get("kernelBody")))?,
                type_scripts: parse_string_array(pow.get("type_scripts").or_else(|| pow.get("typeScripts")))?,
                kernel: parse_string_array(pow.get("kernel"))?,
            });
        } else if pow.is_array() {
            // RPC format: pow is a flat array of paths
            // The RPC format combines all pow paths into one array
            // We need to split them, but we don't know the exact structure
            // As a workaround, distribute paths evenly or put them all in kernel_body
            // This ensures at least some paths are available
            let all_pow_paths = parse_string_array(Some(pow))?;
            if !all_pow_paths.is_empty() {
                // Distribute paths: first third to kernel_body, second third to type_scripts, rest to kernel
                let len = all_pow_paths.len();
                let kernel_body_end = len / 3;
                let type_scripts_end = (len * 2) / 3;
                
                paths.pow = Some(PowPaths {
                    kernel_body: all_pow_paths[..kernel_body_end].to_vec(),
                    type_scripts: all_pow_paths[kernel_body_end..type_scripts_end].to_vec(),
                    kernel: all_pow_paths[type_scripts_end..].to_vec(),
                });
            } else {
                paths.pow = Some(PowPaths {
                    kernel_body: Vec::new(),
                    type_scripts: Vec::new(),
                    kernel: Vec::new(),
                });
            }
        }
    }

    // Parse header section - handle both formats
    if let Some(header) = paths_json.get("header") {
        if header.is_object() {
            // Pool protocol format: { "body": [...], "appendix": [...] }
            paths.header = Some(HeaderPaths {
                body: parse_string_array(header.get("body"))?,
                appendix: parse_string_array(header.get("appendix"))?,
            });
        } else if header.is_array() {
            // RPC format: header is a flat array of paths
            // Split into body and appendix (first half to body, second half to appendix)
            let all_header_paths = parse_string_array(Some(header))?;
            if !all_header_paths.is_empty() {
                let mid = all_header_paths.len() / 2;
                paths.header = Some(HeaderPaths {
                    body: all_header_paths[..mid].to_vec(),
                    appendix: all_header_paths[mid..].to_vec(),
                });
            } else {
                paths.header = Some(HeaderPaths {
                    body: Vec::new(),
                    appendix: Vec::new(),
                });
            }
        }
    }

    // Parse kernel array (same in both formats)
    paths.kernel = parse_string_array(paths_json.get("kernel"))?;

    Ok(paths)
}

fn parse_string_array(value: Option<&serde_json::Value>) -> anyhow::Result<Vec<String>> {
    match value {
        Some(v) if v.is_array() => {
            let mut result = Vec::new();
            for item in v.as_array().unwrap() {
                if let Some(s) = item.as_str() {
                    result.push(s.to_string());
                } else if item.is_array() {
                    // Convert array of numbers to hex string
                    let mut hex = String::new();
                    for num in item.as_array().unwrap() {
                        if let Some(n) = num.as_u64() {
                            hex.push_str(&format!("{:016x}", n));
                        }
                    }
                    result.push(hex);
                }
            }
            Ok(result)
        }
        _ => Ok(Vec::new()),
    }
}

async fn job_fetcher_loop(state: Arc<ServerState>) {
    let interval = state
        .rpc_config
        .as_ref()
        .map(|c| c.fetch_interval_sec)
        .unwrap_or(30);

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;

        if state.rpc_config.is_some() {
            match state.fetch_job_from_rpc().await {
                Ok(Some(new_job)) => {
                    // Check if this is a new job (different ID)
                    let mut current_job = state.current_job.lock().await;
                    let is_new = match current_job.as_ref() {
                        Some(job) => {
                            let current_id = match &job.id {
                                Digest::String(s) => s.clone(),
                                Digest::Array(_) => String::new(),
                            };
                            let new_id = match &new_job.id {
                                Digest::String(s) => s.clone(),
                                Digest::Array(_) => String::new(),
                            };
                            current_id != new_id
                        }
                        None => true,
                    };

                    if is_new {
                        info!("New job fetched from RPC: {:?}", new_job.id);
                        *current_job = Some(new_job.clone());
                        drop(current_job);

                        // Notify all connected clients
                        let clients = state.clients.lock().await;
                        for (worker_id, client) in clients.iter() {
                            if let Some(tx) = &client.writer {
                                let job_notification = json!({
                                    "method": "job",
                                    "params": new_job
                                });
                                let job_json = serde_json::to_string(&job_notification).unwrap();
                                if tx.send(job_json).is_err() {
                                    warn!("Failed to send job notification to worker {}", worker_id);
                                } else {
                                    info!("Sent job notification to worker {}", worker_id);
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    warn!("RPC returned no job (node may be syncing)");
                }
                Err(e) => {
                    error!("Failed to fetch job from RPC: {}", e);
                }
            }
        }
    }
}

async fn handle_client(stream: TcpStream, addr: std::net::SocketAddr, state: Arc<ServerState>) {
    info!("New client connected: {}", addr);
    
    let (reader, writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    let mut worker_id: Option<usize> = None;
    let mut logged_in = false;

    // Create channel for sending job notifications
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Spawn task to send notifications from channel
    // into_split() gives us OwnedReadHalf and OwnedWriteHalf which can be moved
    let mut writer_for_notifications = writer;
    let notification_handle = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if writer_for_notifications.write_all(message.as_bytes()).await.is_err() {
                break;
            }
            if writer_for_notifications.write_all(b"\n").await.is_err() {
                break;
            }
        }
    });

    // Send initial job notification
    {
        let job = state.current_job.lock().await;
        if job.is_none() {
            drop(job);
            let mut job_guard = state.current_job.lock().await;
            if state.rpc_config.is_none() {
                *job_guard = Some(ServerState::create_test_job());
            }
        }
    }

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => {
                info!("Client {} disconnected", addr);
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                info!("Received from {}: {}", addr, trimmed);

                // Parse JSON request
                let request: Result<JsonRequest, _> = serde_json::from_str(trimmed);
                match request {
                    Ok(req) => {
                        let response = match handle_request(req, &state, &mut worker_id, &mut logged_in).await {
                            Ok(resp) => resp,
                            Err(err) => {
                                error!("Error handling request: {:?}", err);
                                JsonResponse::from_err(None, Error::Internal)
                            }
                        };

                        let response_json = serde_json::to_string(&response).unwrap();
                        info!("Sending to {}: {}", addr, response_json);
                        
                        // Send response via the notification channel since writer was moved
                        if tx.send(response_json + "\n").is_err() {
                            error!("Failed to send response");
                            break;
                        }

                        // If login successful, send job notification and store writer channel
                        if logged_in && worker_id.is_some() {
                            // Store writer channel in client state
                            if let Some(wid) = worker_id {
                                let mut clients = state.clients.lock().await;
                                if let Some(client) = clients.get_mut(&wid) {
                                    client.writer = Some(tx.clone());
                                }
                                drop(clients);
                            }

                            if let Some(job) = state.current_job.lock().await.as_ref() {
                                let job_notification = json!({
                                    "method": "job",
                                    "params": job
                                });
                                let job_json = serde_json::to_string(&job_notification).unwrap();
                                info!("Sending job notification to {}: {}", addr, job_json);
                                if tx.send(job_json).is_err() {
                                    error!("Failed to send job notification");
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse request: {} - {}", e, trimmed);
                        let error_response = JsonResponse::from_err(None, Error::Parse);
                        let error_json = serde_json::to_string(&error_response).unwrap();
                        if tx.send(error_json + "\n").is_err() {
                            error!("Failed to send error response");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Error reading from client {}: {}", addr, e);
                break;
            }
        }
    }

    // Cleanup
    if let Some(wid) = worker_id {
        let mut clients = state.clients.lock().await;
        clients.remove(&wid);
        info!("Removed client {} (worker_id: {})", addr, wid);
    }
}

async fn handle_request(
    request: JsonRequest,
    state: &Arc<ServerState>,
    worker_id: &mut Option<usize>,
    logged_in: &mut bool,
) -> std::result::Result<JsonResponse, Error> {
    match request.request {
        Request::Login { name, address, password: _, agent } => {
            info!("Login request: name={}, address={}, agent={}", name, address, agent);
            
            // Generate worker ID
            let mut next_id = state.next_worker_id.lock().await;
            let wid = *next_id;
            *next_id += 1;
            drop(next_id);

            *worker_id = Some(wid);
            *logged_in = true;

            // Store client state
            let mut clients = state.clients.lock().await;
            clients.insert(wid, ClientState {
                worker_id: wid,
                address: address.clone(),
                name: name.clone(),
                agent: agent.clone(),
                connected: true,
                writer: None,
            });
            drop(clients);

            // Get current job
            let job = state.current_job.lock().await.clone();

            Ok(JsonResponse::new(
                request.id,
                Response::Login {
                    id: wid,
                    job: Box::new(job),
                },
            ))
        }
        Request::Keepalived {} => {
            info!("Keepalive from worker {:?}", worker_id);
            Ok(JsonResponse::new(request.id, Response::Keepalived {}))
        }
        Request::Submit { worker, id, pow } => {
            info!("Submit from worker {}: job_id={}, nonce={:?}", worker, id, pow.nonce);
            
            // Check for duplicate
            let mut shares = state.submitted_shares.lock().await;
            let share_key = format!("{}_{:?}", id, pow.nonce);
            if shares.contains(&share_key) {
                warn!("Duplicate share submitted: {}", share_key);
                return Ok(JsonResponse::from_err(request.id, Error::DuplicateShare));
            }
            shares.push(share_key);
            drop(shares);

            // Simple validation - just check nonce is not empty
            if pow.nonce.is_empty() {
                return Ok(JsonResponse::from_err(request.id, Error::LowDifficultyShare));
            }

            info!("Share accepted from worker {}", worker);
            Ok(JsonResponse::new(
                request.id,
                Response::Submit { success: true },
            ))
        }
        Request::Job { .. } => {
            // Job is a notification, shouldn't be in a request with id
            Err(Error::InvalidRequest)
        }
        Request::Pause {} => {
            // Pause is a notification, shouldn't be in a request with id
            Err(Error::InvalidRequest)
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3333".to_string())
        .parse::<u16>()?;

    // Parse RPC configuration from environment
    let rpc_config = if let Ok(rpc_url) = std::env::var("RPC_URL") {
        let rpc_user = std::env::var("RPC_USER").unwrap_or_default();
        let rpc_password = std::env::var("RPC_PASSWORD").unwrap_or_default();
        let guesser_address = std::env::var("GUESSER_ADDRESS")
            .unwrap_or_else(|_| "xnt1test1234567890abcdef".to_string());
        let fetch_interval = std::env::var("RPC_FETCH_INTERVAL")
            .unwrap_or_else(|_| "30".to_string())
            .parse::<u64>()
            .unwrap_or(30);

        info!("RPC configuration:");
        info!("  URL: {}", rpc_url);
        info!("  User: {}", if rpc_user.is_empty() { "(none)" } else { &rpc_user });
        info!("  Guesser Address: {}", guesser_address);
        info!("  Fetch Interval: {}s", fetch_interval);

        Some(RpcConfig {
            url: rpc_url,
            user: rpc_user,
            password: rpc_password,
            guesser_address,
            fetch_interval_sec: fetch_interval,
        })
    } else {
        info!("No RPC_URL configured, using test jobs");
        None
    };

    let state = Arc::new(ServerState::new(rpc_config));
    
    // Initialize with a test job if no RPC configured
    if state.rpc_config.is_none() {
        let mut job = state.current_job.lock().await;
        *job = Some(ServerState::create_test_job());
    } else {
        // Try to fetch initial job from RPC
        match state.fetch_job_from_rpc().await {
            Ok(Some(job)) => {
                info!("Fetched initial job from RPC");
                let mut current_job = state.current_job.lock().await;
                *current_job = Some(job);
            }
            Ok(None) => {
                warn!("RPC returned no initial job, will retry");
            }
            Err(e) => {
                error!("Failed to fetch initial job from RPC: {}", e);
                warn!("Falling back to test job");
                let mut job = state.current_job.lock().await;
                *job = Some(ServerState::create_test_job());
            }
        }
    }

    // Start job fetcher loop if RPC is configured
    if state.rpc_config.is_some() {
        let state_clone = state.clone();
        tokio::spawn(async move {
            job_fetcher_loop(state_clone).await;
        });
    }

    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    info!("Test Stratum Server listening on port {}", port);
    info!("Connect with: stratum+tcp://127.0.0.1:{}", port);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let state_clone = state.clone();
                tokio::spawn(async move {
                    handle_client(stream, addr, state_clone).await;
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
            }
        }
    }
}
