"""
Tests for Friday Orchestrator
=============================

Run with: pytest tests/test_orchestrator.py -v
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from orchestrator.context.detector import ContextDetector
from orchestrator.context.contexts import ContextType, CONTEXTS
from orchestrator.memory.conversation import ConversationMemory, ConversationTurn
from orchestrator.memory.context_builder import (
    ContextBuilder,
    get_default_system_prompt,
)
from orchestrator.tools.registry import ToolRegistry, Tool, ToolResult


class TestContextDetector:
    """Test context detection logic"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_explicit_switch_writers_room(self):
        """Test explicit switch command"""
        context, confidence = self.detector.detect("switch to writers room")
        assert context.context_type == ContextType.WRITERS_ROOM
        assert confidence == 1.0

    def test_explicit_switch_kitchen(self):
        context, confidence = self.detector.detect("go to kitchen mode")
        assert context.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_explicit_switch_storyboard(self):
        context, confidence = self.detector.detect("switch to storyboard")
        assert context.context_type == ContextType.STORYBOARD
        assert confidence == 1.0

    def test_location_detection(self):
        """Test location-based detection"""
        context, confidence = self.detector.detect("hello", location="writers_room")
        assert context.context_type == ContextType.WRITERS_ROOM
        assert confidence == 0.9

        # Reset for fresh test
        self.detector.reset()
        context, confidence = self.detector.detect("hello", location="kitchen")
        assert context.context_type == ContextType.KITCHEN

    def test_keyword_detection_screenplay(self):
        """Test keyword-based detection for screenplay"""
        context, confidence = self.detector.detect("show me scene 5 of the script")
        assert context.context_type == ContextType.WRITERS_ROOM
        assert confidence > 0.5

    def test_keyword_detection_cooking(self):
        """Test keyword-based detection for cooking"""
        self.detector.reset()
        context, confidence = self.detector.detect("how do I cook biryani recipe food")
        assert context.context_type == ContextType.KITCHEN
        assert confidence > 0.5

    def test_keyword_detection_storyboard(self):
        """Test keyword-based detection for storyboard"""
        self.detector.reset()
        context, confidence = self.detector.detect(
            "generate a storyboard frame visual shot"
        )
        assert context.context_type == ContextType.STORYBOARD
        assert confidence > 0.5

    def test_sticky_context(self):
        """Test that context is sticky when no new context detected"""
        # First set writers room explicitly
        context, _ = self.detector.detect("switch to writers room")
        assert context.context_type == ContextType.WRITERS_ROOM

        # Generic message should stick to writers room
        context, confidence = self.detector.detect("what do you think?")
        assert context.context_type == ContextType.WRITERS_ROOM
        assert confidence == 0.6  # Sticky confidence

    def test_default_context(self):
        """Test default when nothing matches (default is WRITERS_ROOM)"""
        self.detector.reset()
        context, confidence = self.detector.detect("hello there")
        # Default context is WRITERS_ROOM per detector init
        assert context.context_type == ContextType.WRITERS_ROOM
        assert confidence == 0.5


class TestConversationMemory:
    """Test conversation memory"""

    def setup_method(self):
        self.memory = ConversationMemory(max_turns=5, max_tokens=1000)

    def test_add_turn(self):
        """Test adding a conversation turn"""
        turn = self.memory.add_turn(
            user_message="Hello Friday",
            assistant_response="Hello Boss, how can I help?",
        )
        assert turn.turn_id == 1
        assert self.memory.turn_count == 1
        assert self.memory.active_turns == 1

    def test_multiple_turns(self):
        """Test multiple turns"""
        for i in range(3):
            self.memory.add_turn(
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
            )
        assert self.memory.turn_count == 3
        assert self.memory.active_turns == 3

    def test_max_turns_limit(self):
        """Test that old turns are dropped when limit exceeded"""
        for i in range(10):
            self.memory.add_turn(
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
            )
        assert self.memory.turn_count == 10
        assert self.memory.active_turns == 5  # max_turns=5

    def test_get_last_n_turns(self):
        """Test getting last N turns"""
        for i in range(5):
            self.memory.add_turn(
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
            )
        last_2 = self.memory.get_last_n_turns(2)
        assert len(last_2) == 2
        assert last_2[-1].user_message == "Message 4"

    def test_context_messages(self):
        """Test building context messages"""
        self.memory.add_turn(
            user_message="Hello",
            assistant_response="Hi Boss",
        )
        messages = self.memory.get_context_messages(system_prompt="You are Friday")
        assert len(messages) >= 2  # System + at least user/assistant
        assert messages[0].role == "system"

    def test_clear(self):
        """Test clearing memory"""
        self.memory.add_turn(
            user_message="Hello",
            assistant_response="Hi",
        )
        self.memory.clear()
        assert self.memory.turn_count == 0
        assert self.memory.active_turns == 0

    def test_serialization(self):
        """Test to_dict and from_dict"""
        self.memory.add_turn(
            user_message="Hello",
            assistant_response="Hi Boss",
        )
        data = self.memory.to_dict()
        restored = ConversationMemory.from_dict(data)
        assert restored.turn_count == 1


class TestToolRegistry:
    """Test tool registry"""

    def setup_method(self):
        self.registry = ToolRegistry()

    def test_builtin_tools_registered(self):
        """Test that built-in tools are registered"""
        tools = self.registry.list_tools()
        tool_names = [t.name for t in tools]

        assert "scene_search" in tool_names
        assert "scene_get" in tool_names
        assert "send_email" in tool_names

    def test_get_tool(self):
        """Test getting a specific tool"""
        tool = self.registry.get("scene_search")
        assert tool is not None
        assert tool.name == "scene_search"
        assert tool.category == "screenplay"

    def test_unknown_tool(self):
        """Test getting unknown tool"""
        tool = self.registry.get("nonexistent_tool")
        assert tool is None

    def test_execute_unknown_tool(self):
        """Test executing unknown tool"""
        result = self.registry.execute("nonexistent_tool", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_openai_format(self):
        """Test OpenAI tool format conversion"""
        tools = self.registry.to_openai_tools(["scene_search"])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "scene_search"

    def test_anthropic_format(self):
        """Test Anthropic tool format conversion"""
        tools = self.registry.to_anthropic_tools(["scene_search"])
        assert len(tools) == 1
        assert tools[0]["name"] == "scene_search"
        assert "input_schema" in tools[0]

    def test_filter_by_names(self):
        """Test filtering tools by name"""
        tools = self.registry.list_tools(["scene_search", "scene_get"])
        assert len(tools) == 2

    def test_custom_tool_registration(self):
        """Test registering custom tool"""

        def handler(x: int) -> ToolResult:
            return ToolResult(success=True, data=x * 2)

        tool = Tool(
            name="double",
            description="Double a number",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=handler,
        )
        self.registry.register(tool)

        result = self.registry.execute("double", {"x": 5})
        assert result.success
        assert result.data == 10


class TestContextBuilder:
    """Test context builder"""

    def setup_method(self):
        self.builder = ContextBuilder(
            base_system_prompt="You are Friday, an AI assistant.",
            max_context_tokens=2000,
        )

    def test_build_basic_context(self):
        """Test building basic context"""
        context = self.builder.build(
            user_message="Hello Friday",
            context_type=ContextType.GENERAL,
        )
        assert len(context.messages) >= 2  # System + user
        assert context.messages[0].role == "system"
        assert context.messages[-1].role == "user"
        assert context.messages[-1].content == "Hello Friday"

    def test_build_with_memory(self):
        """Test building context with conversation memory"""
        memory = ConversationMemory()
        memory.add_turn(
            user_message="Previous message",
            assistant_response="Previous response",
        )

        context = self.builder.build(
            user_message="New message",
            conversation_memory=memory,
            context_type=ContextType.GENERAL,
        )
        # Should have system + history + new user message
        assert len(context.messages) >= 3

    def test_context_specific_prompt(self):
        """Test that context-specific prompts are added"""
        context = self.builder.build(
            user_message="Show me scene 5",
            context_type=ContextType.WRITERS_ROOM,
        )
        system_content = context.messages[0].content
        # Should contain writers room specific content
        assert (
            "screenplay" in system_content.lower() or "scene" in system_content.lower()
        )

    def test_tools_for_context(self):
        """Test that appropriate tools are returned for context"""
        context = self.builder.build(
            user_message="Search for scenes",
            context_type=ContextType.WRITERS_ROOM,
        )
        tool_names = [t["function"]["name"] for t in context.tools]
        assert "scene_search" in tool_names


class TestContextConfigs:
    """Test context configurations"""

    def test_all_contexts_defined(self):
        """Test all context types have configurations"""
        for ctx_type in ContextType:
            assert ctx_type in CONTEXTS

    def test_writers_room_tools(self):
        """Test writers room has screenplay tools"""
        config = CONTEXTS[ContextType.WRITERS_ROOM]
        assert "scene_search" in config.available_tools
        assert "scene_get" in config.available_tools
        assert "scene_update" in config.available_tools

    def test_kitchen_tools(self):
        """Test kitchen has vision tools"""
        config = CONTEXTS[ContextType.KITCHEN]
        assert "camera_analyze" in config.available_tools

    def test_storyboard_tools(self):
        """Test storyboard has image tools"""
        config = CONTEXTS[ContextType.STORYBOARD]
        assert "generate_image" in config.available_tools


class TestDefaultSystemPrompt:
    """Test default system prompt"""

    def test_system_prompt_exists(self):
        """Test that default system prompt is defined"""
        prompt = get_default_system_prompt()
        assert len(prompt) > 100  # Should be substantial

    def test_system_prompt_contains_key_elements(self):
        """Test system prompt contains key personality elements"""
        prompt = get_default_system_prompt()
        assert "Friday" in prompt
        assert "Boss" in prompt
        assert "Telugu" in prompt or "telugu" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
