use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
};
use gaiapi::{Faction, GaiaEnv, GameConfig, Observation, StepResult, rules};
use serde::{Deserialize, Serialize};

#[derive(Default)]
struct ApiState {
    environments: dashmap::DashMap<u64, GaiaEnv>,
    next_id: AtomicU64,
}

type SharedState = Arc<ApiState>;
type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ApiError>)>;

#[derive(Debug, Deserialize)]
struct CreateEnvironmentRequest {
    #[serde(default)]
    config: GameConfig,
}

#[derive(Debug, Deserialize)]
struct StepRequest {
    action: usize,
}

#[derive(Debug, Serialize)]
struct EnvironmentCreated {
    id: u64,
    observation: Observation,
}

#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
}

fn api_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<ApiError>) {
    (
        status,
        Json(ApiError {
            error: message.into(),
        }),
    )
}

async fn create_environment(
    State(state): State<SharedState>,
    Json(request): Json<CreateEnvironmentRequest>,
) -> ApiResult<EnvironmentCreated> {
    let environment = GaiaEnv::new(request.config)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, error.to_string()))?;
    let observation = environment.observe();
    let id = state.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    state.environments.insert(id, environment);
    Ok(Json(EnvironmentCreated { id, observation }))
}

async fn observe_environment(
    State(state): State<SharedState>,
    Path(id): Path<u64>,
) -> ApiResult<Observation> {
    let environment = state.environments.get(&id).ok_or_else(|| {
        api_error(StatusCode::NOT_FOUND, format!("environment {id} was not found"))
    })?;
    Ok(Json(environment.observe()))
}

async fn reset_environment(
    State(state): State<SharedState>,
    Path(id): Path<u64>,
) -> ApiResult<Observation> {
    let mut environment = state.environments.get_mut(&id).ok_or_else(|| {
        api_error(StatusCode::NOT_FOUND, format!("environment {id} was not found"))
    })?;
    Ok(Json(environment.reset()))
}

async fn step_environment(
    State(state): State<SharedState>,
    Path(id): Path<u64>,
    Json(request): Json<StepRequest>,
) -> ApiResult<StepResult> {
    let mut environment = state.environments.get_mut(&id).ok_or_else(|| {
        api_error(StatusCode::NOT_FOUND, format!("environment {id} was not found"))
    })?;
    let cmd = gaiapi::action_space::decode_action(request.action, &environment.map).ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "Invalid action index"))?;
let actor = environment.current_player();
let result = environment.execute_command_from_rl(actor, cmd).map_err(|error| api_error(StatusCode::UNPROCESSABLE_ENTITY, format!("{:?}", error)))?;
    Ok(Json(result))
}

async fn faction_free_actions(Path(faction): Path<Faction>) -> Json<Vec<rules::FreeAction>> {
    Json(rules::free_actions_for(faction))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/environments", post(create_environment))
        .route(
            "/v1/environments/{id}/observation",
            get(observe_environment),
        )
        .route("/v1/environments/{id}/reset", post(reset_environment))
        .route("/v1/environments/{id}/step", post(step_environment))
        .route(
            "/v1/rules/factions/{faction}/free-actions",
            get(faction_free_actions),
        )
        .with_state(Arc::new(ApiState::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("failed to bind gaiapi to 127.0.0.1:3000");
    println!("gaiapi listening on http://127.0.0.1:3000");
    axum::serve(listener, app)
        .await
        .expect("gaiapi server failed");
}



