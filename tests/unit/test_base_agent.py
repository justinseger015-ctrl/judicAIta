"""
Unit tests for BaseAgent class.

Tests cover:
- Agent initialization
- Configuration validation
- Input validation
- Processing execution
- Error handling
- Trace collection
- Logging
- Performance monitoring

Target: >80% code coverage
"""

import logging
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agents.base import (
    AgentConfig,
    AgentResponse,
    AgentStatus,
    BaseAgent,
    ModelFamily,
    TraceStep,
)


# ==================== FIXTURES ====================


@pytest.fixture
def basic_config() -> AgentConfig:
    """Basic agent configuration for testing."""
    return AgentConfig(name="test_agent", model_path="google/gemma-3-1b")


@pytest.fixture
def advanced_config() -> AgentConfig:
    """Advanced agent configuration with all options."""
    return AgentConfig(
        name="advanced_agent",
        model_path="google/gemma-3-1b",
        use_lora=True,
        lora_path="/path/to/lora",
        device="tpu",
        max_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        enable_logging=True,
        enable_tracing=True,
        timeout_seconds=60,
    )


@pytest.fixture
def mock_agent(basic_config: AgentConfig) -> BaseAgent:
    """Mock agent implementation for testing.

    Creates a concrete implementation of BaseAgent with simple
    process and validate_input methods.
    """

    class MockAgent(BaseAgent):
        """Test agent implementation."""

        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Simple process implementation."""
            self.add_trace_step(
                step="mock_process",
                description="Processing mock data",
                input_data=input_data,
            )
            return {"result": "processed", "input_text": input_data.get("text", "")}

        def validate_input(self, input_data: Dict[str, Any]) -> bool:
            """Validate input has required 'text' key."""
            return "text" in input_data

    return MockAgent(basic_config)


@pytest.fixture
def failing_agent(basic_config: AgentConfig) -> BaseAgent:
    """Agent that fails during processing."""

    class FailingAgent(BaseAgent):
        """Agent that raises exception during processing."""

        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Raise exception to test error handling."""
            raise RuntimeError("Intentional failure for testing")

        def validate_input(self, input_data: Dict[str, Any]) -> bool:
            """Always returns True."""
            return True

    return FailingAgent(basic_config)


# ==================== CONFIGURATION TESTS ====================


class TestAgentConfig:
    """Test AgentConfig validation and defaults."""

    def test_minimal_config(self) -> None:
        """Test minimal configuration with defaults."""
        config = AgentConfig(name="minimal_agent")

        assert config.name == "minimal_agent"
        assert config.model_path == "google/gemma-3-1b"
        assert config.use_lora is False
        assert config.device == "cpu"
        assert config.max_tokens == 512
        assert config.temperature == 0.1
        assert config.enable_logging is True
        assert config.enable_tracing is True

    def test_full_config(self, advanced_config: AgentConfig) -> None:
        """Test configuration with all parameters."""
        assert advanced_config.name == "advanced_agent"
        assert advanced_config.use_lora is True
        assert advanced_config.lora_path == "/path/to/lora"
        assert advanced_config.device == "tpu"
        assert advanced_config.max_tokens == 1024
        assert advanced_config.temperature == 0.2
        assert advanced_config.timeout_seconds == 60

    def test_invalid_model_path(self) -> None:
        """Test validation fails for non-Gemma models."""
        with pytest.raises(ValidationError, match="Model must be from Gemma family"):
            AgentConfig(name="test", model_path="openai/gpt-4")

    def test_temperature_validation(self) -> None:
        """Test temperature bounds validation."""
        # Valid temperatures
        AgentConfig(name="test", temperature=0.0)
        AgentConfig(name="test", temperature=1.0)
        AgentConfig(name="test", temperature=2.0)

        # Invalid temperature
        with pytest.raises(ValidationError):
            AgentConfig(name="test", temperature=3.0)

        with pytest.raises(ValidationError):
            AgentConfig(name="test", temperature=-0.1)

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens bounds validation."""
        # Valid values
        AgentConfig(name="test", max_tokens=1)
        AgentConfig(name="test", max_tokens=8192)

        # Invalid values
        with pytest.raises(ValidationError):
            AgentConfig(name="test", max_tokens=0)

        with pytest.raises(ValidationError):
            AgentConfig(name="test", max_tokens=10000)


# ==================== AGENT INITIALIZATION TESTS ====================


class TestBaseAgentInitialization:
    """Test BaseAgent initialization."""

    def test_basic_initialization(self, mock_agent: BaseAgent, basic_config: AgentConfig) -> None:
        """Test agent initializes correctly."""
        assert mock_agent.config == basic_config
        assert mock_agent.status == AgentStatus.IDLE
        assert len(mock_agent._trace) == 0
        assert mock_agent.logger is not None

    def test_logger_setup(self, mock_agent: BaseAgent) -> None:
        """Test logger is configured correctly."""
        assert isinstance(mock_agent.logger, logging.Logger)
        assert mock_agent.logger.name == "judicaita.agents.test_agent"

    def test_logger_disabled(self) -> None:
        """Test logger can be disabled."""

        class TestAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

        config = AgentConfig(name="test", enable_logging=False)
        agent = TestAgent(config)

        assert agent.logger.level == logging.WARNING

    def test_repr(self, mock_agent: BaseAgent) -> None:
        """Test string representation."""
        repr_str = repr(mock_agent)

        assert "MockAgent" in repr_str
        assert "test_agent" in repr_str
        assert "google/gemma-3-1b" in repr_str
        assert "idle" in repr_str


# ==================== TRACE TESTS ====================


class TestReasoningTrace:
    """Test reasoning trace functionality."""

    def test_add_trace_step(self, mock_agent: BaseAgent) -> None:
        """Test adding trace steps."""
        mock_agent.add_trace_step(
            step="step1",
            description="Test step",
            input_data={"key": "value"},
            output_data={"result": "output"},
            duration_ms=10.5,
        )

        trace = mock_agent.get_trace()
        assert len(trace) == 1

        step = trace[0]
        assert step["step"] == "step1"
        assert step["description"] == "Test step"
        assert step["input"] == {"key": "value"}
        assert step["output"] == {"result": "output"}
        assert step["duration_ms"] == 10.5
        assert "timestamp" in step

    def test_trace_disabled(self) -> None:
        """Test trace collection can be disabled."""

        class TestAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                self.add_trace_step("step1", "Test")
                return {}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

        config = AgentConfig(name="test", enable_tracing=False)
        agent = TestAgent(config)
        agent.add_trace_step("step1", "Test")

        assert len(agent.get_trace()) == 0

    def test_clear_trace(self, mock_agent: BaseAgent) -> None:
        """Test clearing trace."""
        mock_agent.add_trace_step("step1", "Test 1")
        mock_agent.add_trace_step("step2", "Test 2")

        assert len(mock_agent.get_trace()) == 2

        mock_agent.clear_trace()

        assert len(mock_agent.get_trace()) == 0

    def test_get_trace_returns_copy(self, mock_agent: BaseAgent) -> None:
        """Test get_trace returns a copy, not reference."""
        mock_agent.add_trace_step("step1", "Test")

        trace1 = mock_agent.get_trace()
        trace2 = mock_agent.get_trace()

        assert trace1 == trace2
        assert trace1 is not trace2  # Different objects


# ==================== EXECUTION TESTS ====================


class TestAgentExecution:
    """Test agent execution flow."""

    def test_successful_execution(self, mock_agent: BaseAgent) -> None:
        """Test successful agent execution."""
        input_data = {"text": "test input"}
        response = mock_agent(input_data)

        assert response.status == AgentStatus.COMPLETED
        assert response.output == {"result": "processed", "input_text": "test input"}
        assert response.error is None
        assert len(response.trace) > 0
        assert response.execution_time_ms > 0
        assert response.model_version == "google/gemma-3-1b"
        assert response.compatibility_score == 1.0

    def test_execution_updates_status(self, mock_agent: BaseAgent) -> None:
        """Test agent status updates during execution."""
        assert mock_agent.status == AgentStatus.IDLE

        mock_agent({"text": "test"})

        assert mock_agent.status == AgentStatus.COMPLETED

    def test_invalid_input_execution(self, mock_agent: BaseAgent) -> None:
        """Test execution with invalid input."""
        input_data = {"wrong_key": "value"}  # Missing 'text' key
        response = mock_agent(input_data)

        assert response.status == AgentStatus.FAILED
        assert "Validation error" in response.error
        assert response.output == {}

    def test_execution_with_exception(self, failing_agent: BaseAgent) -> None:
        """Test execution when process raises exception."""
        input_data = {"text": "test"}
        response = failing_agent(input_data)

        assert response.status == AgentStatus.FAILED
        assert "Intentional failure" in response.error
        assert response.output == {}

    def test_execution_timing(self, mock_agent: BaseAgent) -> None:
        """Test execution time is measured."""

        class SlowAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                time.sleep(0.01)  # 10ms delay
                return {}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

        config = AgentConfig(name="slow_agent")
        agent = SlowAgent(config)

        response = agent({"text": "test"})

        # Should be at least 10ms
        assert response.execution_time_ms >= 10.0

    def test_execution_clears_previous_trace(self, mock_agent: BaseAgent) -> None:
        """Test each execution clears previous trace."""
        # First execution
        response1 = mock_agent({"text": "first"})
        trace1_length = len(response1.trace)

        # Second execution
        response2 = mock_agent({"text": "second"})
        trace2_length = len(response2.trace)

        # Traces should have same structure (not accumulated)
        assert trace1_length == trace2_length


# ==================== COMPATIBILITY TESTS ====================


class TestCrossModelCompatibility:
    """Test cross-model compatibility checking."""

    def test_default_compatibility_score(self, mock_agent: BaseAgent) -> None:
        """Test default compatibility score is 1.0."""
        response = mock_agent({"text": "test"})

        assert response.compatibility_score == 1.0

    def test_custom_compatibility_check(self) -> None:
        """Test custom compatibility checking."""

        class CompatibilityAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

            def _check_compatibility(self) -> float:
                """Custom compatibility check returning 0.95."""
                return 0.95

        config = AgentConfig(name="compat_agent")
        agent = CompatibilityAgent(config)

        response = agent({"text": "test"})

        assert response.compatibility_score == 0.95


# ==================== RESPONSE MODEL TESTS ====================


class TestAgentResponse:
    """Test AgentResponse model."""

    def test_agent_response_creation(self) -> None:
        """Test creating AgentResponse."""
        response = AgentResponse(
            status=AgentStatus.COMPLETED,
            output={"key": "value"},
            model_version="google/gemma-3-1b",
            execution_time_ms=100.5,
        )

        assert response.status == AgentStatus.COMPLETED
        assert response.output == {"key": "value"}
        assert response.metadata == {}
        assert response.trace == []
        assert response.error is None
        assert response.model_version == "google/gemma-3-1b"
        assert response.execution_time_ms == 100.5
        assert response.compatibility_score == 1.0

    def test_agent_response_with_error(self) -> None:
        """Test AgentResponse with error."""
        response = AgentResponse(
            status=AgentStatus.FAILED,
            error="Test error",
            model_version="google/gemma-3-1b",
            execution_time_ms=50.0,
        )

        assert response.status == AgentStatus.FAILED
        assert response.error == "Test error"

    def test_agent_response_timestamp(self) -> None:
        """Test timestamp is auto-generated."""
        response = AgentResponse(
            status=AgentStatus.COMPLETED,
            model_version="google/gemma-3-1b",
            execution_time_ms=10.0,
        )

        # Should have ISO format timestamp
        assert "T" in response.timestamp
        assert len(response.timestamp) > 0


# ==================== EDGE CASES ====================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self) -> None:
        """Test agent with empty input dictionary."""

        class EmptyInputAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"processed": True}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True  # Accept empty input

        config = AgentConfig(name="empty_agent")
        agent = EmptyInputAgent(config)

        response = agent({})

        assert response.status == AgentStatus.COMPLETED

    def test_large_output(self) -> None:
        """Test agent with large output data."""

        class LargeOutputAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Generate large output
                return {"data": "x" * 10000}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

        config = AgentConfig(name="large_agent")
        agent = LargeOutputAgent(config)

        response = agent({"text": "test"})

        assert response.status == AgentStatus.COMPLETED
        assert len(response.output["data"]) == 10000

    def test_none_values_in_input(self, mock_agent: BaseAgent) -> None:
        """Test handling None values in input."""
        input_data = {"text": None}

        # Should fail validation since text is None
        response = mock_agent(input_data)

        # This depends on validate_input implementation
        # In our mock, it checks for 'text' key presence, not value
        assert response.status == AgentStatus.COMPLETED


# ==================== LOGGING TESTS ====================


class TestLogging:
    """Test logging functionality."""

    def test_logging_enabled(self, mock_agent: BaseAgent, caplog) -> None:
        """Test logs are created when logging is enabled."""
        with caplog.at_level(logging.INFO):
            mock_agent({"text": "test"})

        # Check logs were created
        assert len(caplog.records) > 0
        assert any("Processing with test_agent" in record.message for record in caplog.records)

    def test_logging_disabled(self) -> None:
        """Test minimal logging when disabled."""

        class QuietAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

        config = AgentConfig(name="quiet", enable_logging=False)
        agent = QuietAgent(config)

        # Logger should be set to WARNING level
        assert agent.logger.level == logging.WARNING


# ==================== PARAMETRIZED TESTS ====================


@pytest.mark.parametrize(
    "model_path,expected_valid",
    [
        ("google/gemma-3-1b", True),
        ("google/gemma-2.5-1b", True),
        ("google/gemma-3-2b", True),
        ("meta/llama-2-7b", False),
        ("openai/gpt-4", False),
    ],
)
def test_model_path_validation(model_path: str, expected_valid: bool) -> None:
    """Test model path validation with various inputs."""
    if expected_valid:
        config = AgentConfig(name="test", model_path=model_path)
        assert config.model_path == model_path
    else:
        with pytest.raises(ValidationError):
            AgentConfig(name="test", model_path=model_path)


@pytest.mark.parametrize(
    "temperature,expected_valid",
    [
        (0.0, True),
        (0.5, True),
        (1.0, True),
        (2.0, True),
        (-0.1, False),
        (2.5, False),
    ],
)
def test_temperature_bounds(temperature: float, expected_valid: bool) -> None:
    """Test temperature parameter bounds."""
    if expected_valid:
        config = AgentConfig(name="test", temperature=temperature)
        assert config.temperature == temperature
    else:
        with pytest.raises(ValidationError):
            AgentConfig(name="test", temperature=temperature)


# ==================== INTEGRATION-STYLE TESTS ====================


class TestAgentWorkflow:
    """Test complete agent workflows."""

    def test_multi_step_processing(self) -> None:
        """Test agent with multi-step processing."""

        class MultiStepAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Step 1: Parse
                self.add_trace_step("parse", "Parsing input", input_data=input_data)
                parsed = input_data["text"].upper()

                # Step 2: Transform
                self.add_trace_step("transform", "Transforming data")
                transformed = f"Processed: {parsed}"

                # Step 3: Return
                self.add_trace_step("return", "Returning result", output_data=transformed)
                return {"result": transformed}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return "text" in input_data and isinstance(input_data["text"], str)

        config = AgentConfig(name="multi_step", enable_tracing=True)
        agent = MultiStepAgent(config)

        response = agent({"text": "hello"})

        assert response.status == AgentStatus.COMPLETED
        assert response.output["result"] == "Processed: HELLO"
        # Should have at least 3 trace steps plus validation/completion
        assert len(response.trace) >= 5

    def test_error_recovery_workflow(self) -> None:
        """Test workflow with error handling."""

        class RecoveryAgent(BaseAgent):
            def __init__(self, config: AgentConfig):
                super().__init__(config)
                self.retry_count = 0

            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                if input_data.get("should_fail", False):
                    raise ValueError("Simulated failure")
                return {"success": True}

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                return True

        config = AgentConfig(name="recovery")
        agent = RecoveryAgent(config)

        # Test successful case
        response_success = agent({"should_fail": False})
        assert response_success.status == AgentStatus.COMPLETED

        # Test failure case
        response_fail = agent({"should_fail": True})
        assert response_fail.status == AgentStatus.FAILED
        assert "Simulated failure" in response_fail.error
