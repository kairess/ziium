use crate::ast::{BinaryOp, UnaryOp};
use crate::error::{RunError, RuntimeError};
use crate::hir::{self, Expr, Program, SendSelector, Stmt};
use crate::message::{
    KeywordMessage, ResultiveMessage, UnaryMessage, WordMessage, unary_message_for_property,
};
use crate::parser::parse_source_with_metadata;
use crate::resolver::ResolverSession;
use crate::token::Span;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use flate2::read::GzDecoder;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::fs::{self, File};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use tch::nn::{self, Module, OptimizerConfig};
#[cfg(not(target_arch = "wasm32"))]
use tch::{Device, Kind, Tensor, no_grad};

type EnvRef = Rc<RefCell<Environment>>;

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub output: Vec<String>,
    pub canvas_frames: Vec<CanvasFrame>,
    pub events: Vec<ExecutionEvent>,
}

#[derive(Debug)]
pub struct InterpreterSession {
    interpreter: Interpreter,
    resolver: ResolverSession,
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    None,
    List(Rc<RefCell<Vec<Value>>>),
    Record(Rc<RefCell<BTreeMap<String, Value>>>),
    Function(FunctionValue),
    #[cfg(not(target_arch = "wasm32"))]
    Tensor(Rc<Tensor>),
    #[cfg(not(target_arch = "wasm32"))]
    Batch(Rc<TorchBatch>),
    Host(HostValue),
}

#[derive(Debug, Clone)]
pub enum FunctionValue {
    User(UserFunction),
    Builtin(BuiltinFunction),
}

#[derive(Debug, Clone)]
pub struct UserFunction {
    pub name: String,
    pub params: Vec<String>,
    pub body: Rc<Vec<Stmt>>,
    env: EnvRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFunction {
    Length,
    CurrentTimeSeconds,
    Push,
    PopLast,
    ToString,
    ToInt,
    ToFloat,
    FlattenLayer,
    LinearLayer,
    ReluLayer,
    SequentialNetwork,
    BuildMnistDataset,
    BuildDataLoader,
    CrossEntropyLoss,
    AdamOptimizer,
    ZeroGrad,
    Forward,
    ComputeLoss,
    Backward,
    OptimizerStep,
    SetTrainMode,
    SetEvalMode,
    FetchBatch,
    ArgMax,
    CountEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostValue {
    Canvas,
    TorchLayer(TorchLayerKind),
    TorchDataset(usize),
    TorchDataLoader(usize),
    TorchModel(usize),
    TorchModelParameters(usize),
    TorchLossFunction(TorchLossKind),
    TorchOptimizer(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchLayerKind {
    Flatten,
    Relu,
    Linear { in_features: i64, out_features: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchLossKind {
    CrossEntropy,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct TorchDataset {
    images: Tensor,
    labels: Tensor,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct TorchDataLoader {
    dataset: usize,
    batch_size: i64,
    order: Tensor,
    len: i64,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct TorchBatch {
    images: Tensor,
    labels: Tensor,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct TorchModel {
    var_store: nn::VarStore,
    network: nn::Sequential,
    is_training: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct TorchOptimizer {
    optimizer: nn::Optimizer,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CanvasFrame {
    pub commands: Vec<CanvasCommand>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind")]
pub enum ExecutionEvent {
    Output { text: String },
    Sleep { seconds: f64 },
    CanvasFrame { frame: CanvasFrame },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind")]
pub enum CanvasCommand {
    Clear {
        background: String,
    },
    Dot {
        x: f64,
        y: f64,
        color: String,
        size: f64,
    },
    FillRect {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        color: String,
    },
    FillText {
        text: String,
        x: f64,
        y: f64,
        color: String,
        size: f64,
    },
}

#[derive(Debug)]
struct Interpreter {
    globals: EnvRef,
    output: Vec<String>,
    current_canvas_commands: Vec<CanvasCommand>,
    canvas_frames: Vec<CanvasFrame>,
    events: Vec<ExecutionEvent>,
    stream_events: bool,
    #[cfg(not(target_arch = "wasm32"))]
    run_started_at: Instant,
    #[cfg(not(target_arch = "wasm32"))]
    torch_device: Option<Device>,
    #[cfg(not(target_arch = "wasm32"))]
    mnist_cache: Option<tch::vision::dataset::Dataset>,
    #[cfg(not(target_arch = "wasm32"))]
    torch_datasets: Vec<TorchDataset>,
    #[cfg(not(target_arch = "wasm32"))]
    torch_loaders: Vec<TorchDataLoader>,
    #[cfg(not(target_arch = "wasm32"))]
    torch_models: Vec<RefCell<TorchModel>>,
    #[cfg(not(target_arch = "wasm32"))]
    torch_optimizers: Vec<RefCell<TorchOptimizer>>,
}

#[derive(Debug)]
struct Environment {
    values: BTreeMap<String, Value>,
    parent: Option<EnvRef>,
}

#[derive(Debug)]
enum ExecSignal {
    Continue,
    Return(Value),
}

pub fn interpret_program(program: &crate::ast::Program) -> Result<ExecutionResult, RuntimeError> {
    let mut session = InterpreterSession::new();
    session.interpret_program(program)
}

pub fn interpret_hir_program(program: &Program) -> Result<ExecutionResult, RuntimeError> {
    let mut session = InterpreterSession::new();
    session.interpret_hir_program(program)
}

pub fn run_source(source: &str) -> Result<ExecutionResult, RunError> {
    let mut session = InterpreterSession::new();
    session.run_source(source)
}

impl InterpreterSession {
    pub fn new() -> Self {
        Self {
            interpreter: Interpreter::new(),
            resolver: ResolverSession::new(),
        }
    }

    pub fn interpret_program(
        &mut self,
        program: &crate::ast::Program,
    ) -> Result<ExecutionResult, RuntimeError> {
        let hir_program = hir::lower_program(program);
        self.resolver
            .resolve_hir_program(&hir_program)
            .map_err(|err| RuntimeError::with_span(err.message.clone(), err.span.clone()))?;
        self.interpreter.run_program(&hir_program)
    }

    pub fn interpret_hir_program(
        &mut self,
        program: &Program,
    ) -> Result<ExecutionResult, RuntimeError> {
        self.interpreter.run_program(program)
    }

    pub fn run_source(&mut self, source: &str) -> Result<ExecutionResult, RunError> {
        let (program, metadata) = parse_source_with_metadata(source).map_err(RunError::from)?;
        let hir_program = hir::lower_program_with_metadata(&program, Some(&metadata));
        self.resolver
            .resolve_hir_program(&hir_program)
            .map_err(crate::error::FrontendError::from)
            .map_err(RunError::from)?;
        self.interpreter
            .run_program(&hir_program)
            .map_err(RunError::from)
    }

    pub fn set_stream_events(&mut self, enabled: bool) {
        self.interpreter.set_stream_events(enabled);
    }
}

impl Default for InterpreterSession {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    fn new() -> Self {
        let globals = Environment::new(None);
        install_builtins(&globals);

        Self {
            globals,
            output: Vec::new(),
            current_canvas_commands: Vec::new(),
            canvas_frames: Vec::new(),
            events: Vec::new(),
            stream_events: false,
            #[cfg(not(target_arch = "wasm32"))]
            run_started_at: Instant::now(),
            #[cfg(not(target_arch = "wasm32"))]
            torch_device: None,
            #[cfg(not(target_arch = "wasm32"))]
            mnist_cache: None,
            #[cfg(not(target_arch = "wasm32"))]
            torch_datasets: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            torch_loaders: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            torch_models: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            torch_optimizers: Vec::new(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn torch_device(&mut self) -> Device {
        *self.torch_device.get_or_insert_with(select_torch_device)
    }

    fn run_program(&mut self, program: &Program) -> Result<ExecutionResult, RuntimeError> {
        let output_start = self.output.len();
        self.current_canvas_commands.clear();
        self.canvas_frames.clear();
        self.events.clear();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.run_started_at = Instant::now();
        }
        match self.execute_block(&program.statements, self.globals.clone())? {
            ExecSignal::Continue => {
                self.finish_canvas_frame();
                Ok(ExecutionResult {
                    output: self.output[output_start..].to_vec(),
                    canvas_frames: self.canvas_frames.clone(),
                    events: self.events.clone(),
                })
            }
            ExecSignal::Return(_) => Err(RuntimeError::new(
                "`돌려준다`는 함수 본문 안에서만 사용할 수 있습니다.",
            )),
        }
    }

    fn set_stream_events(&mut self, enabled: bool) {
        self.stream_events = enabled;
    }

    fn execute_block(
        &mut self,
        statements: &[Stmt],
        env: EnvRef,
    ) -> Result<ExecSignal, RuntimeError> {
        for statement in statements {
            match self.execute_stmt(statement, env.clone())? {
                ExecSignal::Continue => {}
                signal @ ExecSignal::Return(_) => return Ok(signal),
            }
        }

        Ok(ExecSignal::Continue)
    }

    fn execute_stmt(&mut self, stmt: &Stmt, env: EnvRef) -> Result<ExecSignal, RuntimeError> {
        let stmt_span = stmt.span().cloned();
        match stmt {
            Stmt::Bind { name, value, .. } => {
                if env.borrow().values.contains_key(name) {
                    return Err(RuntimeError::with_span(
                        format!("`{}`은(는) 현재 스코프에 이미 정의되어 있습니다.", name),
                        stmt_span,
                    ));
                }
                let value = self
                    .eval_expr(value, env.clone())
                    .map_err(|err| err.with_fallback_span(stmt_span.clone()))?;
                env.borrow_mut().values.insert(name.clone(), value);
                Ok(ExecSignal::Continue)
            }
            Stmt::Assign { name, value, .. } => {
                let value = self
                    .eval_expr(value, env.clone())
                    .map_err(|err| err.with_fallback_span(stmt_span.clone()))?;
                assign_value(&env, name, value).map_err(|err| err.with_fallback_span(stmt_span))?;
                Ok(ExecSignal::Continue)
            }
            Stmt::Print { value, .. } => {
                let value = self
                    .eval_expr(value, env)
                    .map_err(|err| err.with_fallback_span(stmt_span))?;
                let rendered = value.render();
                if self.stream_events {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let mut stdout = io::stdout().lock();
                        writeln!(stdout, "{rendered}").map_err(|err| {
                            RuntimeError::new(format!("출력을 쓰지 못했습니다: {err}"))
                        })?;
                        stdout.flush().map_err(|err| {
                            RuntimeError::new(format!("출력을 비우지 못했습니다: {err}"))
                        })?;
                    }
                }
                self.output.push(rendered.clone());
                self.events.push(ExecutionEvent::Output { text: rendered });
                Ok(ExecSignal::Continue)
            }
            Stmt::Sleep {
                duration_seconds, ..
            } => {
                let duration_value = self
                    .eval_expr(duration_seconds, env)
                    .map_err(|err| err.with_fallback_span(stmt_span.clone()))?;
                let seconds = expect_sleep_seconds(duration_value)
                    .map_err(|err| err.with_fallback_span(stmt_span))?;
                self.finish_canvas_frame();
                self.events.push(ExecutionEvent::Sleep { seconds });
                if self.stream_events {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        thread::sleep(Duration::from_secs_f64(seconds));
                    }
                }
                Ok(ExecSignal::Continue)
            }
            Stmt::Return { value, .. } => {
                let value = self
                    .eval_expr(value, env)
                    .map_err(|err| err.with_fallback_span(stmt_span))?;
                Ok(ExecSignal::Return(value))
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let condition_span = condition.span().cloned().or_else(|| stmt_span.clone());
                let condition = self
                    .eval_expr(condition, env.clone())
                    .map_err(|err| err.with_fallback_span(condition_span.clone()))?;
                match condition {
                    Value::Bool(true) => self.execute_block(then_block, env),
                    Value::Bool(false) => {
                        if let Some(else_block) = else_block {
                            self.execute_block(else_block, env)
                        } else {
                            Ok(ExecSignal::Continue)
                        }
                    }
                    _ => Err(RuntimeError::with_span(
                        "조건식은 `참` 또는 `거짓`이어야 합니다.",
                        condition_span,
                    )),
                }
            }
            Stmt::While {
                condition, body, ..
            } => {
                let condition_span = condition.span().cloned().or_else(|| stmt_span.clone());
                loop {
                    let condition_value = self
                        .eval_expr(condition, env.clone())
                        .map_err(|err| err.with_fallback_span(condition_span.clone()))?;
                    match condition_value {
                        Value::Bool(true) => match self.execute_block(body, env.clone())? {
                            ExecSignal::Continue => {}
                            signal @ ExecSignal::Return(_) => return Ok(signal),
                        },
                        Value::Bool(false) => break,
                        _ => {
                            return Err(RuntimeError::with_span(
                                "반복 조건식은 `참` 또는 `거짓`이어야 합니다.",
                                condition_span,
                            ));
                        }
                    }
                }
                Ok(ExecSignal::Continue)
            }
            Stmt::FunctionDef {
                name, params, body, ..
            } => {
                if env.borrow().values.contains_key(name) {
                    return Err(RuntimeError::with_span(
                        format!("`{}`은(는) 현재 스코프에 이미 정의되어 있습니다.", name),
                        stmt_span,
                    ));
                }

                let cloned_body = Rc::new(body.clone());
                let function = Value::Function(FunctionValue::User(UserFunction {
                    name: name.clone(),
                    params: params.clone(),
                    body: cloned_body.clone(),
                    env: env.clone(),
                }));
                env.borrow_mut().values.insert(name.clone(), function);
                Ok(ExecSignal::Continue)
            }
            Stmt::Send {
                receiver,
                selector,
                args,
                ..
            } => {
                let receiver = self
                    .eval_expr(receiver, env.clone())
                    .map_err(|err| err.with_fallback_span(stmt_span.clone()))?;
                let mut arg_values = Vec::with_capacity(args.len());
                for arg in args {
                    arg_values.push(
                        self.eval_expr(arg, env.clone())
                            .map_err(|err| err.with_fallback_span(stmt_span.clone()))?,
                    );
                }
                self.execute_send_stmt(receiver, selector, arg_values)
                    .map_err(|err| err.with_fallback_span(stmt_span))?;
                Ok(ExecSignal::Continue)
            }
            Stmt::NamedCall {
                callee, named_args, ..
            } => {
                let callee = self
                    .eval_expr(callee, env.clone())
                    .map_err(|err| err.with_fallback_span(stmt_span.clone()))?;
                let named_args = self
                    .eval_expr(named_args, env)
                    .map_err(|err| err.with_fallback_span(stmt_span.clone()))?;
                self.call_named_value(callee, named_args, stmt_span)?;
                Ok(ExecSignal::Continue)
            }
            Stmt::Expr { expr, .. } => {
                self.eval_expr(expr, env)
                    .map_err(|err| err.with_fallback_span(stmt_span))?;
                Ok(ExecSignal::Continue)
            }
        }
    }

    fn eval_expr(&mut self, expr: &Expr, env: EnvRef) -> Result<Value, RuntimeError> {
        let expr_span = expr.span().cloned();
        match expr {
            Expr::Name { name, .. } => {
                lookup_value(&env, name).map_err(|err| err.with_fallback_span(expr_span))
            }
            Expr::Int { raw, .. } => raw.parse::<i64>().map(Value::Int).map_err(|_| {
                RuntimeError::with_span(
                    format!("정수 리터럴 `{}`를 해석할 수 없습니다.", raw),
                    expr_span,
                )
            }),
            Expr::Float { raw, .. } => raw.parse::<f64>().map(Value::Float).map_err(|_| {
                RuntimeError::with_span(
                    format!("실수 리터럴 `{}`를 해석할 수 없습니다.", raw),
                    expr_span,
                )
            }),
            Expr::String { value, .. } => Ok(Value::String(value.clone())),
            Expr::Bool { value, .. } => Ok(Value::Bool(*value)),
            Expr::None { .. } => Ok(Value::None),
            Expr::List { items, .. } => {
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values.push(
                        self.eval_expr(item, env.clone())
                            .map_err(|err| err.with_fallback_span(expr_span.clone()))?,
                    );
                }
                Ok(Value::List(Rc::new(RefCell::new(values))))
            }
            Expr::Record { entries, .. } => {
                let mut map = BTreeMap::new();
                for entry in entries {
                    map.insert(
                        entry.key.clone(),
                        self.eval_expr(&entry.value, env.clone())
                            .map_err(|err| err.with_fallback_span(expr_span.clone()))?,
                    );
                }
                Ok(Value::Record(Rc::new(RefCell::new(map))))
            }
            Expr::Unary { op, expr, .. } => {
                let value = self
                    .eval_expr(expr, env)
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                self.eval_unary(*op, value)
                    .map_err(|err| err.with_fallback_span(expr_span))
            }
            Expr::Binary {
                left, op, right, ..
            } => {
                let left = self
                    .eval_expr(left, env.clone())
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                let right = self
                    .eval_expr(right, env)
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                self.eval_binary(left, *op, right)
                    .map_err(|err| err.with_fallback_span(expr_span))
            }
            Expr::Call { callee, args, .. } => {
                let callee = self
                    .eval_expr(callee, env.clone())
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                let mut arg_values = Vec::with_capacity(args.len());
                for arg in args {
                    arg_values.push(
                        self.eval_expr(arg, env.clone())
                            .map_err(|err| err.with_fallback_span(expr_span.clone()))?,
                    );
                }
                self.call_value(callee, arg_values, expr_span.clone())
                    .map_err(|err| err.with_fallback_span(expr_span))
            }
            Expr::Send {
                receiver,
                selector,
                args,
                ..
            } => {
                let receiver = self
                    .eval_expr(receiver, env.clone())
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                let mut arg_values = Vec::with_capacity(args.len());
                for arg in args {
                    arg_values.push(
                        self.eval_expr(arg, env.clone())
                            .map_err(|err| err.with_fallback_span(expr_span.clone()))?,
                    );
                }
                self.eval_send_expr(receiver, selector, arg_values, env, expr_span.clone())
                    .map_err(|err| err.with_fallback_span(expr_span))
            }
            Expr::Index { base, index, .. } => {
                let base = self
                    .eval_expr(base, env.clone())
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                let index = self
                    .eval_expr(index, env)
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                self.eval_index(base, index)
                    .map_err(|err| err.with_fallback_span(expr_span))
            }
        }
    }

    fn eval_unary(&self, op: UnaryOp, value: Value) -> Result<Value, RuntimeError> {
        match op {
            UnaryOp::Negate => match value {
                Value::Int(value) => Ok(Value::Int(-value)),
                Value::Float(value) => Ok(Value::Float(-value)),
                _ => Err(RuntimeError::new("단항 `-`는 숫자에만 사용할 수 있습니다.")),
            },
            UnaryOp::Not => match value {
                Value::Bool(value) => Ok(Value::Bool(!value)),
                _ => Err(RuntimeError::new(
                    "`아니다`는 불리언 값에만 사용할 수 있습니다.",
                )),
            },
        }
    }

    fn eval_binary(&self, left: Value, op: BinaryOp, right: Value) -> Result<Value, RuntimeError> {
        match op {
            BinaryOp::Add => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{a}{b}"))),
                _ => Err(RuntimeError::new(
                    "`+`는 숫자끼리 또는 문자열끼리만 사용할 수 있습니다.",
                )),
            },
            BinaryOp::Subtract => numeric_binary(left, right, |a, b| a - b, |a, b| a - b),
            BinaryOp::Multiply => numeric_binary(left, right, |a, b| a * b, |a, b| a * b),
            BinaryOp::Divide => numeric_binary(left, right, |a, b| a / b, |a, b| a / b),
            BinaryOp::Modulo => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                _ => Err(RuntimeError::new("`%`는 정수끼리만 사용할 수 있습니다.")),
            },
            BinaryOp::Equal => Ok(Value::Bool(values_equal(&left, &right))),
            BinaryOp::NotEqual => Ok(Value::Bool(!values_equal(&left, &right))),
            BinaryOp::Less => comparison_binary(left, right, |a, b| a < b),
            BinaryOp::LessEqual => comparison_binary(left, right, |a, b| a <= b),
            BinaryOp::Greater => comparison_binary(left, right, |a, b| a > b),
            BinaryOp::GreaterEqual => comparison_binary(left, right, |a, b| a >= b),
            BinaryOp::And => match (left, right) {
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
                _ => Err(RuntimeError::new(
                    "`그리고`는 불리언 값에만 사용할 수 있습니다.",
                )),
            },
            BinaryOp::Or => match (left, right) {
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
                _ => Err(RuntimeError::new(
                    "`또는`은 불리언 값에만 사용할 수 있습니다.",
                )),
            },
        }
    }

    fn call_value(
        &mut self,
        callee: Value,
        args: Vec<Value>,
        call_span: Option<Span>,
    ) -> Result<Value, RuntimeError> {
        match callee {
            Value::Function(FunctionValue::Builtin(function)) => self
                .call_builtin(function, args)
                .map_err(|err| err.with_call_frame(function.name(), call_span)),
            Value::Function(FunctionValue::User(function)) => {
                if function.params.len() != args.len() {
                    return Err(RuntimeError::new(format!(
                        "함수 인수 개수가 맞지 않습니다. 기대: {}, 실제: {}",
                        function.params.len(),
                        args.len()
                    )));
                }

                let frame = Environment::new(Some(function.env.clone()));
                for (param, arg) in function.params.iter().zip(args) {
                    frame.borrow_mut().values.insert(param.clone(), arg);
                }

                match self
                    .execute_block(function.body.as_ref(), frame)
                    .map_err(|err| err.with_call_frame(function.name.clone(), call_span))?
                {
                    ExecSignal::Continue => Ok(Value::None),
                    ExecSignal::Return(value) => Ok(value),
                }
            }
            _ => Err(RuntimeError::new("호출할 수 없는 값을 호출했습니다.")),
        }
    }

    fn call_named_value(
        &mut self,
        callee: Value,
        named_args: Value,
        call_span: Option<Span>,
    ) -> Result<Value, RuntimeError> {
        match callee {
            Value::Function(FunctionValue::User(function)) => {
                let named_args = expect_record("호출한다", named_args)?;

                for key in named_args.keys() {
                    if !function.params.iter().any(|param| param == key) {
                        return Err(RuntimeError::new(format!(
                            "`{}` 함수에는 `{}` 인수가 없습니다.",
                            function.name, key
                        )));
                    }
                }

                let mut ordered_args = Vec::with_capacity(function.params.len());
                for param in &function.params {
                    let value = named_args.get(param).cloned().ok_or_else(|| {
                        RuntimeError::new(format!(
                            "`{}` 함수 호출에 `{}` 인수가 필요합니다.",
                            function.name, param
                        ))
                    })?;
                    ordered_args.push(value);
                }

                self.call_value(
                    Value::Function(FunctionValue::User(function)),
                    ordered_args,
                    call_span,
                )
            }
            Value::Function(FunctionValue::Builtin(function)) => Err(RuntimeError::new(format!(
                "`{}`에는 아직 이름 붙은 호출을 사용할 수 없습니다.",
                function.name()
            ))),
            _ => Err(RuntimeError::new("호출할 수 없는 값을 호출했습니다.")),
        }
    }

    fn call_builtin(
        &mut self,
        function: BuiltinFunction,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        match function {
            BuiltinFunction::Length => {
                let [value] = expect_arity::<1>("길이", args)?;
                match value {
                    Value::List(items) => Ok(Value::Int(items.borrow().len() as i64)),
                    Value::String(text) => Ok(Value::Int(text.chars().count() as i64)),
                    Value::Record(map) => Ok(Value::Int(map.borrow().len() as i64)),
                    _ => Err(RuntimeError::new(
                        "`길이`는 목록, 문자열, 레코드에만 사용할 수 있습니다.",
                    )),
                }
            }
            BuiltinFunction::CurrentTimeSeconds => {
                let [] = expect_arity::<0>("현재시간초", args)?;
                #[cfg(target_arch = "wasm32")]
                {
                    Err(RuntimeError::new(
                        "현재 환경에서는 `현재시간초`를 사용할 수 없습니다.",
                    ))
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Ok(Value::Float(self.run_started_at.elapsed().as_secs_f64()))
                }
            }
            BuiltinFunction::Push => {
                let [list, value] = expect_arity::<2>("추가", args)?;
                match list {
                    Value::List(items) => {
                        items.borrow_mut().push(value);
                        Ok(Value::None)
                    }
                    _ => Err(RuntimeError::new(
                        "`추가`의 첫 번째 인수는 목록이어야 합니다.",
                    )),
                }
            }
            BuiltinFunction::PopLast => {
                let [list] = expect_arity::<1>("마지막꺼내기", args)?;
                match list {
                    Value::List(items) => items.borrow_mut().pop().ok_or_else(|| {
                        RuntimeError::new("빈 목록에서는 마지막 값을 꺼낼 수 없습니다.")
                    }),
                    _ => Err(RuntimeError::new(
                        "`마지막꺼내기`는 목록에만 사용할 수 있습니다.",
                    )),
                }
            }
            BuiltinFunction::ToString => {
                let [value] = expect_arity::<1>("문자열로", args)?;
                Ok(Value::String(value.render()))
            }
            BuiltinFunction::ToInt => {
                let [value] = expect_arity::<1>("정수로", args)?;
                match value {
                    Value::Int(value) => Ok(Value::Int(value)),
                    Value::Float(value) => Ok(Value::Int(value as i64)),
                    Value::String(value) => value
                        .parse::<i64>()
                        .map(Value::Int)
                        .map_err(|_| RuntimeError::new("문자열을 정수로 바꿀 수 없습니다.")),
                    _ => Err(RuntimeError::new(
                        "`정수로`는 숫자 또는 문자열에만 사용할 수 있습니다.",
                    )),
                }
            }
            BuiltinFunction::ToFloat => {
                let [value] = expect_arity::<1>("실수로", args)?;
                match value {
                    Value::Int(value) => Ok(Value::Float(value as f64)),
                    Value::Float(value) => Ok(Value::Float(value)),
                    Value::String(value) => value
                        .parse::<f64>()
                        .map(Value::Float)
                        .map_err(|_| RuntimeError::new("문자열을 실수로 바꿀 수 없습니다.")),
                    _ => Err(RuntimeError::new(
                        "`실수로`는 숫자 또는 문자열에만 사용할 수 있습니다.",
                    )),
                }
            }
            BuiltinFunction::FlattenLayer => {
                let [] = expect_arity::<0>("평탄화", args)?;
                Ok(Value::Host(HostValue::TorchLayer(TorchLayerKind::Flatten)))
            }
            BuiltinFunction::LinearLayer => {
                let [in_features, out_features] = expect_arity::<2>("선형층", args)?;
                let in_features = expect_int("선형층", in_features)?;
                let out_features = expect_int("선형층", out_features)?;
                Ok(Value::Host(HostValue::TorchLayer(TorchLayerKind::Linear {
                    in_features,
                    out_features,
                })))
            }
            BuiltinFunction::ReluLayer => {
                let [] = expect_arity::<0>("렐루", args)?;
                Ok(Value::Host(HostValue::TorchLayer(TorchLayerKind::Relu)))
            }
            BuiltinFunction::SequentialNetwork => {
                let [layers] = expect_arity::<1>("순차신경망", args)?;
                self.build_sequential_network(layers)
            }
            BuiltinFunction::BuildMnistDataset => {
                let [config] = expect_arity::<1>("숫자손글씨데이터셋", args)?;
                self.build_mnist_dataset(config)
            }
            BuiltinFunction::BuildDataLoader => {
                let [config] = expect_arity::<1>("데이터로더", args)?;
                self.build_data_loader(config)
            }
            BuiltinFunction::CrossEntropyLoss => {
                let [] = expect_arity::<0>("교차엔트로피손실", args)?;
                Ok(Value::Host(HostValue::TorchLossFunction(
                    TorchLossKind::CrossEntropy,
                )))
            }
            BuiltinFunction::AdamOptimizer => {
                let [parameters, learning_rate] = expect_arity::<2>("아담", args)?;
                self.build_adam_optimizer(parameters, learning_rate)
            }
            BuiltinFunction::ZeroGrad => {
                let [optimizer] = expect_arity::<1>("기울기초기화", args)?;
                self.zero_grad_optimizer(optimizer)
            }
            BuiltinFunction::Forward => {
                let [model, input] = expect_arity::<2>("순전파", args)?;
                self.forward_model(model, input)
            }
            BuiltinFunction::ComputeLoss => {
                let [loss_fn, logits, labels] = expect_arity::<3>("손실계산", args)?;
                self.compute_loss(loss_fn, logits, labels)
            }
            BuiltinFunction::Backward => {
                let [loss] = expect_arity::<1>("역전파", args)?;
                self.backward_tensor(loss)
            }
            BuiltinFunction::OptimizerStep => {
                let [optimizer] = expect_arity::<1>("매개변수갱신", args)?;
                self.step_optimizer(optimizer)
            }
            BuiltinFunction::SetTrainMode => {
                let [model] = expect_arity::<1>("학습모드로바꾸기", args)?;
                self.set_model_mode(model, true)
            }
            BuiltinFunction::SetEvalMode => {
                let [model] = expect_arity::<1>("평가모드로바꾸기", args)?;
                self.set_model_mode(model, false)
            }
            BuiltinFunction::FetchBatch => {
                let [loader, batch_index] = expect_arity::<2>("배치가져오기", args)?;
                self.fetch_batch(loader, batch_index)
            }
            BuiltinFunction::ArgMax => {
                let [tensor, dim] = expect_arity::<2>("최대인덱스", args)?;
                self.argmax_tensor(tensor, dim)
            }
            BuiltinFunction::CountEqual => {
                let [left, right] = expect_arity::<2>("같은값개수", args)?;
                self.count_equal(left, right)
            }
        }
    }

    fn build_sequential_network(&mut self, layers: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = layers;
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let layer_values = expect_list_values("순차신경망", layers)?;
            let mut kinds = Vec::with_capacity(layer_values.len());
            for value in layer_values {
                match value {
                    Value::Host(HostValue::TorchLayer(kind)) => kinds.push(kind),
                    _ => {
                        return Err(RuntimeError::new(
                            "`순차신경망`에는 레이어 목록만 전달할 수 있습니다.",
                        ));
                    }
                }
            }

            let device = self.torch_device();
            let var_store = nn::VarStore::new(device);
            let path = var_store.root();
            let mut network = nn::seq();
            for (index, kind) in kinds.into_iter().enumerate() {
                match kind {
                    TorchLayerKind::Flatten => {
                        network = network.add_fn(|xs| xs.flatten(1, -1));
                    }
                    TorchLayerKind::Relu => {
                        network = network.add_fn(|xs| xs.relu());
                    }
                    TorchLayerKind::Linear {
                        in_features,
                        out_features,
                    } => {
                        let name = format!("linear{index}");
                        network = network.add(nn::linear(
                            &path / name,
                            in_features,
                            out_features,
                            Default::default(),
                        ));
                    }
                }
            }

            Ok(self.register_model(TorchModel {
                var_store,
                network,
                is_training: true,
            }))
        }
    }

    fn build_mnist_dataset(&mut self, config: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = config;
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let record = expect_record("숫자손글씨데이터셋", config)?;
            let is_train = expect_bool_field(&record, "학습용")?;
            let device = self.torch_device();
            if self.mnist_cache.is_none() {
                self.mnist_cache = Some(load_mnist_bundle()?);
            }
            let (images, labels) = {
                let mnist = self.mnist_cache.as_ref().unwrap();
                if is_train {
                    (mnist.train_images.shallow_clone(), mnist.train_labels.shallow_clone())
                } else {
                    (mnist.test_images.shallow_clone(), mnist.test_labels.shallow_clone())
                }
            };
            Ok(self.register_dataset(TorchDataset {
                images: images.to_device(device).to_kind(Kind::Float),
                labels: labels.to_device(device).to_kind(Kind::Int64),
            }))
        }
    }

    fn build_data_loader(&mut self, config: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = config;
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let record = expect_record("데이터로더", config)?;
            let dataset_id = expect_host_id_field(&record, "데이터셋", |value| match value {
                Value::Host(HostValue::TorchDataset(id)) => Some(*id),
                _ => None,
            })?;
            let batch_size = expect_int_field(&record, "배치크기")?;
            if batch_size <= 0 {
                return Err(RuntimeError::new("`배치크기`는 1 이상의 정수여야 합니다."));
            }
            let shuffle = expect_bool_field(&record, "섞기여부")?;
            let device = self.torch_device();
            let dataset = self
                .torch_datasets
                .get(dataset_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 데이터셋입니다."))?;
            let count = dataset.labels.size().first().copied().unwrap_or(0);
            let len = (count + batch_size - 1) / batch_size;
            let order = if shuffle {
                Tensor::randperm(count, (Kind::Int64, device))
            } else {
                Tensor::arange(count, (Kind::Int64, device))
            };

            Ok(self.register_loader(TorchDataLoader {
                dataset: dataset_id,
                batch_size,
                order,
                len,
            }))
        }
    }

    fn build_adam_optimizer(
        &mut self,
        parameters: Value,
        learning_rate: Value,
    ) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (parameters, learning_rate);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model_id = match parameters {
                Value::Host(HostValue::TorchModelParameters(id))
                | Value::Host(HostValue::TorchModel(id)) => id,
                _ => {
                    return Err(RuntimeError::new(
                        "`아담`의 첫 번째 인수는 모델의 `매개변수`여야 합니다.",
                    ));
                }
            };
            let learning_rate = expect_float("아담", learning_rate)?;
            if learning_rate <= 0.0 {
                return Err(RuntimeError::new("학습률은 0보다 커야 합니다."));
            }

            let model = self
                .torch_models
                .get(model_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 모델입니다."))?;
            let model_ref = model.borrow();
            let optimizer = nn::Adam::default()
                .build(&model_ref.var_store, learning_rate)
                .map_err(|err| RuntimeError::new(format!("아담 최적화기 생성 실패: {err}")))?;
            drop(model_ref);
            Ok(self.register_optimizer(TorchOptimizer { optimizer }))
        }
    }

    fn zero_grad_optimizer(&mut self, optimizer: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = optimizer;
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let optimizer_id = expect_optimizer_id(optimizer)?;
            let optimizer = self
                .torch_optimizers
                .get(optimizer_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 최적화기입니다."))?;
            optimizer.borrow_mut().optimizer.zero_grad();
            Ok(Value::None)
        }
    }

    fn forward_model(&mut self, model: Value, input: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (model, input);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model_id = expect_model_id(model)?;
            let input_tensor = expect_tensor("순전파", input)?;
            let model_cell = self
                .torch_models
                .get(model_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 모델입니다."))?;
            let model = model_cell.borrow();
            let logits = if model.is_training {
                model.network.forward(&input_tensor)
            } else {
                no_grad(|| model.network.forward(&input_tensor))
            };
            Ok(Self::wrap_tensor(logits))
        }
    }

    fn compute_loss(
        &mut self,
        loss_fn: Value,
        logits: Value,
        labels: Value,
    ) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (loss_fn, logits, labels);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let kind = match loss_fn {
                Value::Host(HostValue::TorchLossFunction(kind)) => kind,
                _ => {
                    return Err(RuntimeError::new(
                        "`손실계산`의 첫 번째 인수는 손실함수여야 합니다.",
                    ));
                }
            };
            let logits = expect_tensor("손실계산", logits)?;
            let labels = expect_tensor("손실계산", labels)?;
            let loss = match kind {
                TorchLossKind::CrossEntropy => logits.cross_entropy_for_logits(&labels),
            };
            Ok(Self::wrap_tensor(loss))
        }
    }

    fn backward_tensor(&mut self, loss: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = loss;
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let loss = expect_tensor("역전파", loss)?;
            loss.backward();
            Ok(Value::None)
        }
    }

    fn step_optimizer(&mut self, optimizer: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = optimizer;
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let optimizer_id = expect_optimizer_id(optimizer)?;
            let optimizer = self
                .torch_optimizers
                .get(optimizer_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 최적화기입니다."))?;
            optimizer.borrow_mut().optimizer.step();
            Ok(Value::None)
        }
    }

    fn set_model_mode(&mut self, model: Value, is_training: bool) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (model, is_training);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model_id = expect_model_id(model)?;
            let model = self
                .torch_models
                .get(model_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 모델입니다."))?;
            model.borrow_mut().is_training = is_training;
            Ok(Value::None)
        }
    }

    fn fetch_batch(&mut self, loader: Value, batch_index: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (loader, batch_index);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let loader_id = expect_loader_id(loader)?;
            let batch_index = expect_int("배치가져오기", batch_index)?;
            if batch_index < 0 {
                return Err(RuntimeError::new("배치 인덱스는 0 이상의 정수여야 합니다."));
            }
            let loader = self
                .torch_loaders
                .get(loader_id)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 데이터로더입니다."))?;
            if batch_index >= loader.len {
                return Err(RuntimeError::new("배치 인덱스가 범위를 벗어났습니다."));
            }
            let dataset = self
                .torch_datasets
                .get(loader.dataset)
                .ok_or_else(|| RuntimeError::new("유효하지 않은 데이터셋입니다."))?;
            let start = batch_index * loader.batch_size;
            let total = dataset.labels.size().first().copied().unwrap_or(0);
            let count = (total - start).min(loader.batch_size);
            let batch_indices = loader.order.narrow(0, start, count);

            let images = dataset.images.index_select(0, &batch_indices);
            let labels = dataset.labels.index_select(0, &batch_indices);

            Ok(Self::wrap_batch(TorchBatch { images, labels }))
        }
    }

    fn argmax_tensor(&mut self, tensor: Value, dim: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (tensor, dim);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let tensor = expect_tensor("최대인덱스", tensor)?;
            let dim = expect_int("최대인덱스", dim)?;
            Ok(Self::wrap_tensor(tensor.argmax(dim, false)))
        }
    }

    fn count_equal(&mut self, left: Value, right: Value) -> Result<Value, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (left, right);
            self.pytorch_unavailable()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let left = expect_tensor("같은값개수", left)?;
            let right = expect_tensor("같은값개수", right)?;
            let count = left.eq_tensor(&right).sum(Kind::Int64).int64_value(&[]);
            Ok(Value::Int(count))
        }
    }

    fn eval_index(&self, base: Value, index: Value) -> Result<Value, RuntimeError> {
        let index = match index {
            Value::Int(value) if value >= 0 => value as usize,
            _ => return Err(RuntimeError::new("인덱스는 0 이상의 정수여야 합니다.")),
        };

        match base {
            Value::List(items) => items
                .borrow()
                .get(index)
                .cloned()
                .ok_or_else(|| RuntimeError::new("목록 인덱스가 범위를 벗어났습니다.")),
            Value::String(text) => text
                .chars()
                .nth(index)
                .map(|ch| Value::String(ch.to_string()))
                .ok_or_else(|| RuntimeError::new("문자열 인덱스가 범위를 벗어났습니다.")),
            _ => Err(RuntimeError::new(
                "인덱싱은 목록 또는 문자열에만 사용할 수 있습니다.",
            )),
        }
    }

    fn eval_word_message(
        &self,
        receiver: Value,
        selector: WordMessage,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        let [arg] = expect_arity::<1>(selector.name(), args)?;
        match selector {
            WordMessage::Add => self.eval_binary(receiver, BinaryOp::Add, arg),
            WordMessage::Subtract => self.eval_binary(receiver, BinaryOp::Subtract, arg),
            WordMessage::Multiply => self.eval_binary(receiver, BinaryOp::Multiply, arg),
            WordMessage::Divide => self.eval_binary(receiver, BinaryOp::Divide, arg),
        }
    }

    fn eval_send_expr(
        &mut self,
        receiver: Value,
        selector: &SendSelector,
        args: Vec<Value>,
        env: EnvRef,
        expr_span: Option<Span>,
    ) -> Result<Value, RuntimeError> {
        match selector {
            SendSelector::Property(name) => {
                if !args.is_empty() {
                    return Err(RuntimeError::new("속성 메시지는 인수를 받을 수 없습니다."));
                }
                self.eval_property(receiver, name)
            }
            SendSelector::Transform(callee_name) => {
                if !args.is_empty() {
                    return Err(RuntimeError::new(
                        "변환 호출은 추가 인수를 받을 수 없습니다.",
                    ));
                }
                let callee = lookup_value(&env, callee_name)
                    .map_err(|err| err.with_fallback_span(expr_span.clone()))?;
                self.call_value(callee, vec![receiver], expr_span)
            }
            SendSelector::Word(selector) => self.eval_word_message(receiver, *selector, args),
            SendSelector::Resultive(selector) => {
                if !args.is_empty() {
                    return Err(RuntimeError::new(
                        "결과 서술 메시지는 추가 인수를 받을 수 없습니다.",
                    ));
                }
                self.send_resultive_message(receiver, *selector)
            }
            SendSelector::Keyword(_) => Err(RuntimeError::new(
                "키워드 메시지는 표현식 자리에서 사용할 수 없습니다.",
            )),
        }
    }

    fn execute_send_stmt(
        &mut self,
        receiver: Value,
        selector: &SendSelector,
        args: Vec<Value>,
    ) -> Result<(), RuntimeError> {
        match selector {
            SendSelector::Keyword(selector) => self.send_keyword_message(receiver, *selector, args),
            SendSelector::Resultive(selector) => {
                if !args.is_empty() {
                    return Err(RuntimeError::new(
                        "결과 서술 메시지는 추가 인수를 받을 수 없습니다.",
                    ));
                }
                self.send_resultive_message(receiver, *selector).map(|_| ())
            }
            _ => Err(RuntimeError::new(
                "이 메시지는 문장 자리에서 사용할 수 없습니다.",
            )),
        }
    }

    fn eval_property(&mut self, base: Value, name: &str) -> Result<Value, RuntimeError> {
        match base {
            Value::Record(map) => {
                if let Some(value) = map.borrow().get(name).cloned() {
                    return Ok(value);
                }

                if let Some(selector) = unary_message_for_property(name) {
                    return self.send_unary_message(Value::Record(map), selector);
                }

                Err(RuntimeError::new(format!(
                    "이 값에는 `{}` 속성이 없습니다.",
                    name
                )))
            }
            #[cfg(not(target_arch = "wasm32"))]
            Value::Batch(batch) => match name {
                "이미지들" => Ok(Self::wrap_tensor(batch.images.shallow_clone())),
                "라벨들" => Ok(Self::wrap_tensor(batch.labels.shallow_clone())),
                _ => Err(RuntimeError::new(format!(
                    "배치에는 `{}` 속성이 없습니다.",
                    name
                ))),
            },
            Value::Host(HostValue::TorchModel(model_id)) => match name {
                "매개변수" => Ok(Value::Host(HostValue::TorchModelParameters(model_id))),
                _ => Err(RuntimeError::new(format!(
                    "모델에는 `{}` 속성이 없습니다.",
                    name
                ))),
            },
            #[cfg(not(target_arch = "wasm32"))]
            Value::Tensor(tensor) => {
                let tensor = tensor.as_ref();
                match name {
                    "길이" => {
                        let size = tensor.size();
                        Ok(Value::Int(size.first().copied().unwrap_or(0)))
                    }
                    "값" => {
                        if tensor.numel() != 1 {
                            return Err(RuntimeError::new(
                                "`값` 속성은 스칼라 텐서에서만 읽을 수 있습니다.",
                            ));
                        }
                        Ok(Value::Float(tensor.double_value(&[])))
                    }
                    _ => Err(RuntimeError::new(format!(
                        "텐서에는 `{}` 속성이 없습니다.",
                        name
                    ))),
                }
            }
            other => {
                if let Some(selector) = unary_message_for_property(name) {
                    self.send_unary_message(other, selector)
                } else {
                    Err(RuntimeError::new(format!(
                        "이 값에는 `{}` 속성이 없습니다.",
                        name
                    )))
                }
            }
        }
    }

    fn send_unary_message(
        &self,
        receiver: Value,
        selector: UnaryMessage,
    ) -> Result<Value, RuntimeError> {
        match (receiver, selector) {
            (Value::List(items), UnaryMessage::Length) => {
                Ok(Value::Int(items.borrow().len() as i64))
            }
            (Value::String(text), UnaryMessage::Length) => {
                Ok(Value::Int(text.chars().count() as i64))
            }
            (Value::Record(map), UnaryMessage::Length) => Ok(Value::Int(map.borrow().len() as i64)),
            (Value::Host(HostValue::TorchDataLoader(loader_id)), UnaryMessage::Length) => {
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = loader_id;
                    Err(RuntimeError::new(
                        "현재 환경에서는 PyTorch 래퍼를 사용할 수 없습니다.",
                    ))
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let loader = self
                        .torch_loaders
                        .get(loader_id)
                        .ok_or_else(|| RuntimeError::new("유효하지 않은 데이터로더입니다."))?;
                    Ok(Value::Int(loader.len))
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            (Value::Tensor(tensor), UnaryMessage::Length) => {
                let size = tensor.size();
                Ok(Value::Int(size.first().copied().unwrap_or(0)))
            }
            (Value::Int(value), UnaryMessage::Square) => Ok(Value::Int(value * value)),
            (Value::Float(value), UnaryMessage::Square) => Ok(Value::Float(value * value)),
            (Value::List(_), UnaryMessage::Square) => {
                Err(RuntimeError::new("목록에는 `제곱` 속성이 없습니다."))
            }
            (Value::String(_), UnaryMessage::Square) => {
                Err(RuntimeError::new("문자열에는 `제곱` 속성이 없습니다."))
            }
            (Value::Record(_), UnaryMessage::Square) => {
                Err(RuntimeError::new("이 값에는 `제곱` 속성이 없습니다."))
            }
            (Value::Int(_), UnaryMessage::Length) => {
                Err(RuntimeError::new("정수에는 `길이` 속성이 없습니다."))
            }
            (Value::Float(_), UnaryMessage::Length) => {
                Err(RuntimeError::new("실수에는 `길이` 속성이 없습니다."))
            }
            (_, UnaryMessage::Length) => {
                Err(RuntimeError::new("이 값에는 `길이` 속성이 없습니다."))
            }
            (_, UnaryMessage::Square) => {
                Err(RuntimeError::new("이 값에는 `제곱` 속성이 없습니다."))
            }
        }
    }

    fn send_resultive_message(
        &self,
        receiver: Value,
        selector: ResultiveMessage,
    ) -> Result<Value, RuntimeError> {
        match selector {
            ResultiveMessage::PopTopElement | ResultiveMessage::PopBackElement => match receiver {
                Value::List(items) => items.borrow_mut().pop().ok_or_else(|| {
                    RuntimeError::new(format!(
                        "빈 목록에서는 {}를 꺼낼 수 없습니다.",
                        selector.role()
                    ))
                }),
                _ => Err(RuntimeError::new(format!(
                    "`{}를 꺼낸` 결과 서술은 목록에만 사용할 수 있습니다.",
                    selector.role()
                ))),
            },
            ResultiveMessage::PopFrontElement => match receiver {
                Value::List(items) => {
                    let mut items = items.borrow_mut();
                    if items.is_empty() {
                        Err(RuntimeError::new(format!(
                            "빈 목록에서는 {}를 꺼낼 수 없습니다.",
                            selector.role()
                        )))
                    } else {
                        Ok(items.remove(0))
                    }
                }
                _ => Err(RuntimeError::new(format!(
                    "`{}를 꺼낸` 결과 서술은 목록에만 사용할 수 있습니다.",
                    selector.role()
                ))),
            },
        }
    }

    fn send_keyword_message(
        &mut self,
        receiver: Value,
        selector: KeywordMessage,
        args: Vec<Value>,
    ) -> Result<(), RuntimeError> {
        match (receiver, selector) {
            (Value::List(items), KeywordMessage::Push) => {
                let [arg] = expect_arity::<1>("추가", args)?;
                items.borrow_mut().push(arg);
                Ok(())
            }
            (Value::Host(HostValue::Canvas), selector) => {
                let [arg] = expect_arity::<1>("그림판 메시지", args)?;
                self.execute_canvas_message(selector, arg)
            }
            (_, KeywordMessage::Push) => Err(RuntimeError::new(
                "`추가` 메시지는 목록에만 보낼 수 있습니다.",
            )),
            (_, KeywordMessage::CanvasClear) => Err(RuntimeError::new(
                "`지우기` 메시지는 그림판에만 보낼 수 있습니다.",
            )),
            (_, KeywordMessage::CanvasFillRect) => Err(RuntimeError::new(
                "`사각형채우기` 메시지는 그림판에만 보낼 수 있습니다.",
            )),
            (_, KeywordMessage::CanvasFillText) => Err(RuntimeError::new(
                "`글자쓰기` 메시지는 그림판에만 보낼 수 있습니다.",
            )),
            (_, KeywordMessage::CanvasDot) => Err(RuntimeError::new(
                "`점찍기` 메시지는 그림판에만 보낼 수 있습니다.",
            )),
        }
    }

    fn execute_canvas_message(
        &mut self,
        selector: KeywordMessage,
        arg: Value,
    ) -> Result<(), RuntimeError> {
        let record = expect_record(selector.name(), arg)?;
        match selector {
            KeywordMessage::CanvasClear => {
                let background = expect_string_field(&record, &["배경색", "색"])?;
                self.begin_canvas_frame();
                self.current_canvas_commands
                    .push(CanvasCommand::Clear { background });
                Ok(())
            }
            KeywordMessage::CanvasDot => {
                let x = expect_number_field(&record, "x")?;
                let y = expect_number_field(&record, "y")?;
                let color = expect_string_field(&record, &["색"])?;
                let size = expect_number_field_or(record.get("크기"), 8.0, "크기")?;
                self.current_canvas_commands
                    .push(CanvasCommand::Dot { x, y, color, size });
                Ok(())
            }
            KeywordMessage::CanvasFillRect => {
                let x = expect_number_field(&record, "x")?;
                let y = expect_number_field(&record, "y")?;
                let width = expect_number_field(&record, "너비")?;
                let height = expect_number_field(&record, "높이")?;
                let color = expect_string_field(&record, &["색"])?;
                self.current_canvas_commands.push(CanvasCommand::FillRect {
                    x,
                    y,
                    width,
                    height,
                    color,
                });
                Ok(())
            }
            KeywordMessage::CanvasFillText => {
                let text = expect_string_field(&record, &["글"])?;
                let x = expect_number_field(&record, "x")?;
                let y = expect_number_field(&record, "y")?;
                let color = expect_string_field(&record, &["색"])?;
                let size = expect_number_field(&record, "크기")?;
                self.current_canvas_commands.push(CanvasCommand::FillText {
                    text,
                    x,
                    y,
                    color,
                    size,
                });
                Ok(())
            }
            KeywordMessage::Push => Err(RuntimeError::new(
                "`그림판`은 `추가` 동작을 지원하지 않습니다.",
            )),
        }
    }

    fn begin_canvas_frame(&mut self) {
        if !self.current_canvas_commands.is_empty() {
            self.finish_canvas_frame();
        }
    }

    fn finish_canvas_frame(&mut self) {
        if self.current_canvas_commands.is_empty() {
            return;
        }

        let frame = CanvasFrame {
            commands: std::mem::take(&mut self.current_canvas_commands),
        };
        self.canvas_frames.push(frame.clone());
        self.events.push(ExecutionEvent::CanvasFrame { frame });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wrap_tensor(tensor: Tensor) -> Value {
        Value::Tensor(Rc::new(tensor))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn register_dataset(&mut self, dataset: TorchDataset) -> Value {
        self.torch_datasets.push(dataset);
        Value::Host(HostValue::TorchDataset(self.torch_datasets.len() - 1))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn register_loader(&mut self, loader: TorchDataLoader) -> Value {
        self.torch_loaders.push(loader);
        Value::Host(HostValue::TorchDataLoader(self.torch_loaders.len() - 1))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wrap_batch(batch: TorchBatch) -> Value {
        Value::Batch(Rc::new(batch))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn register_model(&mut self, model: TorchModel) -> Value {
        self.torch_models.push(RefCell::new(model));
        Value::Host(HostValue::TorchModel(self.torch_models.len() - 1))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn register_optimizer(&mut self, optimizer: TorchOptimizer) -> Value {
        self.torch_optimizers.push(RefCell::new(optimizer));
        Value::Host(HostValue::TorchOptimizer(self.torch_optimizers.len() - 1))
    }

    #[cfg(target_arch = "wasm32")]
    fn pytorch_unavailable<T>(&self) -> Result<T, RuntimeError> {
        Err(RuntimeError::new(
            "현재 환경에서는 PyTorch 래퍼를 사용할 수 없습니다.",
        ))
    }
}

impl Environment {
    fn new(parent: Option<EnvRef>) -> EnvRef {
        Rc::new(RefCell::new(Self {
            values: BTreeMap::new(),
            parent,
        }))
    }
}

impl Value {
    pub fn render(&self) -> String {
        match self {
            Value::Int(value) => value.to_string(),
            Value::Float(value) => {
                if value.fract() == 0.0 {
                    format!("{value:.1}")
                } else {
                    value.to_string()
                }
            }
            Value::Bool(true) => "참".to_string(),
            Value::Bool(false) => "거짓".to_string(),
            Value::String(value) => value.clone(),
            Value::None => "없음".to_string(),
            Value::List(values) => {
                let items = values
                    .borrow()
                    .iter()
                    .map(Value::render)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{items}]")
            }
            Value::Record(values) => {
                let items = values
                    .borrow()
                    .iter()
                    .map(|(key, value)| format!("{key}: {}", value.render()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{ {items} }}")
            }
            Value::Function(FunctionValue::Builtin(function)) => function.to_string(),
            Value::Function(FunctionValue::User(_)) => "<함수>".to_string(),
            Value::Host(HostValue::Canvas) => "<그림판>".to_string(),
            Value::Host(HostValue::TorchLayer(TorchLayerKind::Flatten)) => {
                "<레이어 평탄화>".to_string()
            }
            Value::Host(HostValue::TorchLayer(TorchLayerKind::Relu)) => "<레이어 렐루>".to_string(),
            Value::Host(HostValue::TorchLayer(TorchLayerKind::Linear { .. })) => {
                "<레이어 선형층>".to_string()
            }
            Value::Host(HostValue::TorchDataset(_)) => "<MNIST 데이터셋>".to_string(),
            Value::Host(HostValue::TorchDataLoader(_)) => "<데이터로더>".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            Value::Batch(_) => "<배치>".to_string(),
            Value::Host(HostValue::TorchModel(_)) => "<신경망 모델>".to_string(),
            Value::Host(HostValue::TorchModelParameters(_)) => "<모델 매개변수>".to_string(),
            Value::Host(HostValue::TorchLossFunction(_)) => "<손실함수>".to_string(),
            Value::Host(HostValue::TorchOptimizer(_)) => "<최적화기>".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            Value::Tensor(_) => "<텐서>".to_string(),
        }
    }
}

impl fmt::Display for BuiltinFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuiltinFunction::Length => write!(f, "<내장 함수 길이>"),
            BuiltinFunction::CurrentTimeSeconds => write!(f, "<내장 함수 현재시간초>"),
            BuiltinFunction::Push => write!(f, "<내장 함수 추가>"),
            BuiltinFunction::PopLast => write!(f, "<내장 함수 마지막꺼내기>"),
            BuiltinFunction::ToString => write!(f, "<내장 함수 문자열로>"),
            BuiltinFunction::ToInt => write!(f, "<내장 함수 정수로>"),
            BuiltinFunction::ToFloat => write!(f, "<내장 함수 실수로>"),
            BuiltinFunction::FlattenLayer => write!(f, "<내장 함수 평탄화>"),
            BuiltinFunction::LinearLayer => write!(f, "<내장 함수 선형층>"),
            BuiltinFunction::ReluLayer => write!(f, "<내장 함수 렐루>"),
            BuiltinFunction::SequentialNetwork => write!(f, "<내장 함수 순차신경망>"),
            BuiltinFunction::BuildMnistDataset => write!(f, "<내장 함수 숫자손글씨데이터셋>"),
            BuiltinFunction::BuildDataLoader => write!(f, "<내장 함수 데이터로더>"),
            BuiltinFunction::CrossEntropyLoss => write!(f, "<내장 함수 교차엔트로피손실>"),
            BuiltinFunction::AdamOptimizer => write!(f, "<내장 함수 아담>"),
            BuiltinFunction::ZeroGrad => write!(f, "<내장 함수 기울기초기화>"),
            BuiltinFunction::Forward => write!(f, "<내장 함수 순전파>"),
            BuiltinFunction::ComputeLoss => write!(f, "<내장 함수 손실계산>"),
            BuiltinFunction::Backward => write!(f, "<내장 함수 역전파>"),
            BuiltinFunction::OptimizerStep => write!(f, "<내장 함수 매개변수갱신>"),
            BuiltinFunction::SetTrainMode => write!(f, "<내장 함수 학습모드로바꾸기>"),
            BuiltinFunction::SetEvalMode => write!(f, "<내장 함수 평가모드로바꾸기>"),
            BuiltinFunction::FetchBatch => write!(f, "<내장 함수 배치가져오기>"),
            BuiltinFunction::ArgMax => write!(f, "<내장 함수 최대인덱스>"),
            BuiltinFunction::CountEqual => write!(f, "<내장 함수 같은값개수>"),
        }
    }
}

impl BuiltinFunction {
    fn name(self) -> &'static str {
        match self {
            BuiltinFunction::Length => "길이",
            BuiltinFunction::CurrentTimeSeconds => "현재시간초",
            BuiltinFunction::Push => "추가",
            BuiltinFunction::PopLast => "마지막꺼내기",
            BuiltinFunction::ToString => "문자열로",
            BuiltinFunction::ToInt => "정수로",
            BuiltinFunction::ToFloat => "실수로",
            BuiltinFunction::FlattenLayer => "평탄화",
            BuiltinFunction::LinearLayer => "선형층",
            BuiltinFunction::ReluLayer => "렐루",
            BuiltinFunction::SequentialNetwork => "순차신경망",
            BuiltinFunction::BuildMnistDataset => "숫자손글씨데이터셋",
            BuiltinFunction::BuildDataLoader => "데이터로더",
            BuiltinFunction::CrossEntropyLoss => "교차엔트로피손실",
            BuiltinFunction::AdamOptimizer => "아담",
            BuiltinFunction::ZeroGrad => "기울기초기화",
            BuiltinFunction::Forward => "순전파",
            BuiltinFunction::ComputeLoss => "손실계산",
            BuiltinFunction::Backward => "역전파",
            BuiltinFunction::OptimizerStep => "매개변수갱신",
            BuiltinFunction::SetTrainMode => "학습모드로바꾸기",
            BuiltinFunction::SetEvalMode => "평가모드로바꾸기",
            BuiltinFunction::FetchBatch => "배치가져오기",
            BuiltinFunction::ArgMax => "최대인덱스",
            BuiltinFunction::CountEqual => "같은값개수",
        }
    }
}

fn install_builtins(env: &EnvRef) {
    let mut env = env.borrow_mut();
    env.values.insert(
        "길이".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::Length)),
    );
    env.values.insert(
        "현재시간초".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::CurrentTimeSeconds)),
    );
    env.values.insert(
        "추가".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::Push)),
    );
    env.values.insert(
        "마지막꺼내기".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::PopLast)),
    );
    env.values.insert(
        "문자열로".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ToString)),
    );
    env.values.insert(
        "정수로".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ToInt)),
    );
    env.values.insert(
        "실수로".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ToFloat)),
    );
    env.values.insert(
        "평탄화".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::FlattenLayer)),
    );
    env.values.insert(
        "선형층".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::LinearLayer)),
    );
    env.values.insert(
        "렐루".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ReluLayer)),
    );
    env.values.insert(
        "순차신경망".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::SequentialNetwork)),
    );
    env.values.insert(
        "숫자손글씨데이터셋".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::BuildMnistDataset)),
    );
    // Backward compatibility alias for older samples.
    env.values.insert(
        "숫자손글씨데이터셋만들기".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::BuildMnistDataset)),
    );
    env.values.insert(
        "데이터로더".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::BuildDataLoader)),
    );
    // Backward compatibility alias for older samples.
    env.values.insert(
        "데이터로더만들기".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::BuildDataLoader)),
    );
    env.values.insert(
        "교차엔트로피손실".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::CrossEntropyLoss)),
    );
    env.values.insert(
        "아담".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::AdamOptimizer)),
    );
    env.values.insert(
        "기울기초기화".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ZeroGrad)),
    );
    env.values.insert(
        "순전파".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::Forward)),
    );
    env.values.insert(
        "손실계산".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ComputeLoss)),
    );
    env.values.insert(
        "역전파".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::Backward)),
    );
    env.values.insert(
        "매개변수갱신".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::OptimizerStep)),
    );
    // Backward compatibility alias for previous naming.
    env.values.insert(
        "스텝".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::OptimizerStep)),
    );
    env.values.insert(
        "학습모드로바꾸기".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::SetTrainMode)),
    );
    env.values.insert(
        "평가모드로바꾸기".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::SetEvalMode)),
    );
    env.values.insert(
        "배치가져오기".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::FetchBatch)),
    );
    env.values.insert(
        "최대인덱스".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::ArgMax)),
    );
    env.values.insert(
        "같은값개수".into(),
        Value::Function(FunctionValue::Builtin(BuiltinFunction::CountEqual)),
    );
    env.values
        .insert("그림판".into(), Value::Host(HostValue::Canvas));
}

fn lookup_value(env: &EnvRef, name: &str) -> Result<Value, RuntimeError> {
    let mut current = Some(env.clone());
    while let Some(scope) = current {
        let scope_ref = scope.borrow();
        if let Some(value) = scope_ref.values.get(name) {
            return Ok(value.clone());
        }
        current = scope_ref.parent.clone();
    }

    Err(RuntimeError::new(format!(
        "`{}`은(는) 아직 정의되지 않았습니다.",
        name
    )))
}

fn assign_value(env: &EnvRef, name: &str, value: Value) -> Result<(), RuntimeError> {
    let mut current = Some(env.clone());
    while let Some(scope) = current {
        let parent = {
            let mut scope_ref = scope.borrow_mut();
            if scope_ref.values.contains_key(name) {
                scope_ref.values.insert(name.to_string(), value);
                return Ok(());
            }
            scope_ref.parent.clone()
        };
        current = parent;
    }

    Err(RuntimeError::new(format!(
        "`{}`를 바꿀 수 없습니다. 이 이름이 현재 스코프에 없습니다.",
        name
    )))
}

fn expect_int(name: &str, value: Value) -> Result<i64, RuntimeError> {
    match value {
        Value::Int(value) => Ok(value),
        _ => Err(RuntimeError::new(format!(
            "`{name}` 인수는 정수여야 합니다."
        ))),
    }
}

fn expect_float(name: &str, value: Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Int(value) => Ok(value as f64),
        Value::Float(value) => Ok(value),
        _ => Err(RuntimeError::new(format!(
            "`{name}` 인수는 숫자여야 합니다."
        ))),
    }
}

fn expect_model_id(value: Value) -> Result<usize, RuntimeError> {
    match value {
        Value::Host(HostValue::TorchModel(id)) => Ok(id),
        _ => Err(RuntimeError::new("이 값은 모델이어야 합니다.")),
    }
}

fn expect_optimizer_id(value: Value) -> Result<usize, RuntimeError> {
    match value {
        Value::Host(HostValue::TorchOptimizer(id)) => Ok(id),
        _ => Err(RuntimeError::new("이 값은 최적화기여야 합니다.")),
    }
}

fn expect_loader_id(value: Value) -> Result<usize, RuntimeError> {
    match value {
        Value::Host(HostValue::TorchDataLoader(id)) => Ok(id),
        _ => Err(RuntimeError::new("이 값은 데이터로더여야 합니다.")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn expect_tensor(name: &str, value: Value) -> Result<Rc<Tensor>, RuntimeError> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        _ => Err(RuntimeError::new(format!(
            "`{name}` 인수는 텐서여야 합니다."
        ))),
    }
}

fn expect_list_values(name: &str, value: Value) -> Result<Vec<Value>, RuntimeError> {
    match value {
        Value::List(items) => Ok(items.borrow().clone()),
        _ => Err(RuntimeError::new(format!(
            "`{name}` 인수는 목록이어야 합니다."
        ))),
    }
}

fn expect_bool_field(record: &BTreeMap<String, Value>, key: &str) -> Result<bool, RuntimeError> {
    match record.get(key) {
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(RuntimeError::new(format!(
            "`{key}` 필드는 불리언이어야 합니다."
        ))),
        None => Err(RuntimeError::new(format!("`{key}` 필드가 필요합니다."))),
    }
}

fn expect_int_field(record: &BTreeMap<String, Value>, key: &str) -> Result<i64, RuntimeError> {
    match record.get(key) {
        Some(Value::Int(value)) => Ok(*value),
        Some(_) => Err(RuntimeError::new(format!(
            "`{key}` 필드는 정수여야 합니다."
        ))),
        None => Err(RuntimeError::new(format!("`{key}` 필드가 필요합니다."))),
    }
}

fn expect_host_id_field(
    record: &BTreeMap<String, Value>,
    key: &str,
    extract: impl Fn(&Value) -> Option<usize>,
) -> Result<usize, RuntimeError> {
    match record.get(key) {
        Some(value) => extract(value)
            .ok_or_else(|| RuntimeError::new(format!("`{key}` 필드 타입이 올바르지 않습니다."))),
        None => Err(RuntimeError::new(format!("`{key}` 필드가 필요합니다."))),
    }
}

fn numeric_binary(
    left: Value,
    right: Value,
    int_op: impl FnOnce(i64, i64) -> i64,
    float_op: impl FnOnce(f64, f64) -> f64,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(a, b as f64))),
        _ => Err(RuntimeError::new(
            "숫자 연산은 정수 또는 실수에만 사용할 수 있습니다.",
        )),
    }
}

fn comparison_binary(
    left: Value,
    right: Value,
    op: impl FnOnce(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(op(a as f64, b as f64))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(op(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Bool(op(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(op(a, b as f64))),
        _ => Err(RuntimeError::new(
            "비교 연산은 숫자에만 사용할 수 있습니다.",
        )),
    }
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Int(a), Value::Float(b)) => (*a as f64) == *b,
        (Value::Float(a), Value::Int(b)) => *a == (*b as f64),
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::None, Value::None) => true,
        #[cfg(not(target_arch = "wasm32"))]
        (Value::Tensor(a), Value::Tensor(b)) => Rc::ptr_eq(a, b),
        #[cfg(not(target_arch = "wasm32"))]
        (Value::Batch(a), Value::Batch(b)) => Rc::ptr_eq(a, b),
        (Value::Host(a), Value::Host(b)) => a == b,
        (Value::List(a), Value::List(b)) => {
            let a = a.borrow();
            let b = b.borrow();
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(a, b)| values_equal(a, b))
        }
        (Value::Record(a), Value::Record(b)) => {
            let a = a.borrow();
            let b = b.borrow();
            a.len() == b.len()
                && a.iter().all(|(key, a_value)| {
                    b.get(key)
                        .is_some_and(|b_value| values_equal(a_value, b_value))
                })
        }
        _ => false,
    }
}

fn expect_record(name: &str, value: Value) -> Result<BTreeMap<String, Value>, RuntimeError> {
    match value {
        Value::Record(map) => Ok(map.borrow().clone()),
        _ => Err(RuntimeError::new(format!(
            "`{name}` 인수는 레코드여야 합니다."
        ))),
    }
}

fn expect_number_field(record: &BTreeMap<String, Value>, key: &str) -> Result<f64, RuntimeError> {
    match record.get(key) {
        Some(Value::Int(value)) => Ok(*value as f64),
        Some(Value::Float(value)) => Ok(*value),
        Some(_) => Err(RuntimeError::new(format!(
            "`{key}` 필드는 숫자여야 합니다."
        ))),
        None => Err(RuntimeError::new(format!("`{key}` 필드가 필요합니다."))),
    }
}

fn expect_number_field_or(
    value: Option<&Value>,
    default: f64,
    key: &str,
) -> Result<f64, RuntimeError> {
    match value {
        Some(Value::Int(value)) => Ok(*value as f64),
        Some(Value::Float(value)) => Ok(*value),
        Some(_) => Err(RuntimeError::new(format!(
            "`{key}` 필드는 숫자여야 합니다."
        ))),
        None => Ok(default),
    }
}

fn expect_string_field(
    record: &BTreeMap<String, Value>,
    keys: &[&str],
) -> Result<String, RuntimeError> {
    for key in keys {
        if let Some(value) = record.get(*key) {
            return match value {
                Value::String(value) => Ok(value.clone()),
                _ => Err(RuntimeError::new(format!(
                    "`{key}` 필드는 문자열이어야 합니다."
                ))),
            };
        }
    }

    Err(RuntimeError::new(format!(
        "`{}` 필드가 필요합니다.",
        keys.join("` 또는 `")
    )))
}

fn expect_sleep_seconds(value: Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Int(value) if value >= 0 => Ok(value as f64),
        Value::Float(value) if value.is_finite() && value >= 0.0 => Ok(value),
        Value::Int(_) | Value::Float(_) => {
            Err(RuntimeError::new("`쉬기` 시간은 0 이상의 값이어야 합니다."))
        }
        _ => Err(RuntimeError::new("`쉬기` 시간은 숫자여야 합니다.")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn select_torch_device() -> Device {
    if !tch::Cuda::is_available() {
        return Device::Cpu;
    }
    let cuda = Device::Cuda(0);
    if Tensor::f_zeros([1i64], (Kind::Float, cuda)).is_ok() {
        cuda
    } else {
        Device::Cpu
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_mnist_bundle() -> Result<tch::vision::dataset::Dataset, RuntimeError> {
    let download_dir = Path::new("data/MNIST/raw");
    let last_error = if download_dir.exists() {
        tch::vision::mnist::load_dir(download_dir)
            .err()
            .map(|err| (download_dir.display().to_string(), err))
    } else {
        None
    };

    download_mnist_files(download_dir).map_err(|download_err| {
        let download_dir = download_dir.display();
        match &last_error {
            Some((dir, err)) => RuntimeError::new(format!(
                "MNIST 데이터를 `{download_dir}`에서 불러오지 못해 자동 다운로드를 시도했지만 실패했습니다. 마지막 실패: `{dir}` - {err}, 다운로드 오류: {download_err}"
            )),
            None => RuntimeError::new(format!(
                "MNIST 데이터가 `{download_dir}`에 없어 자동 다운로드를 시도했지만 실패했습니다. 다운로드 오류: {download_err}"
            )),
        }
    })?;

    tch::vision::mnist::load_dir(download_dir).map_err(|err| {
        RuntimeError::new(format!(
            "MNIST 데이터를 자동 다운로드한 뒤에도 불러오지 못했습니다. 대상 경로: `{}` - {err}",
            download_dir.display()
        ))
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn download_mnist_files(dir: &Path) -> Result<(), RuntimeError> {
    const BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
    const FILES: [&str; 4] = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ];

    fs::create_dir_all(dir).map_err(|err| {
        RuntimeError::new(format!(
            "MNIST 저장 디렉터리를 만들지 못했습니다. `{}` - {err}",
            dir.display()
        ))
    })?;

    let client = reqwest::blocking::Client::builder()
        .build()
        .map_err(|err| RuntimeError::new(format!("MNIST 다운로드 클라이언트 생성 실패: {err}")))?;

    for file_name in FILES {
        let destination = dir.join(file_name);
        if destination.exists() {
            continue;
        }

        let url = format!("{BASE_URL}/{file_name}.gz");
        let response = client
            .get(&url)
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|err| {
                RuntimeError::new(format!(
                    "MNIST 파일을 내려받지 못했습니다. `{file_name}` ({url}) - {err}"
                ))
            })?;
        let bytes = response.bytes().map_err(|err| {
            RuntimeError::new(format!(
                "MNIST 다운로드 응답을 읽지 못했습니다. `{file_name}` ({url}) - {err}"
            ))
        })?;

        let mut decoder = GzDecoder::new(bytes.as_ref());
        let mut output = File::create(&destination).map_err(|err| {
            RuntimeError::new(format!(
                "MNIST 파일을 저장하지 못했습니다. `{}` - {err}",
                destination.display()
            ))
        })?;
        std::io::copy(&mut decoder, &mut output).map_err(|err| {
            RuntimeError::new(format!(
                "MNIST 압축 해제에 실패했습니다. `{file_name}` - {err}"
            ))
        })?;
    }

    Ok(())
}

fn expect_arity<const N: usize>(name: &str, args: Vec<Value>) -> Result<[Value; N], RuntimeError> {
    args.try_into().map_err(|values: Vec<Value>| {
        RuntimeError::new(format!(
            "`{}` 함수 인수 개수가 맞지 않습니다. 기대: {}, 실제: {}",
            name,
            N,
            values.len()
        ))
    })
}
