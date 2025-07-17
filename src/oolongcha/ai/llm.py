from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import logging
from uuid import uuid4
import uuid
import re

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageState(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    ERROR = "error"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    response: Optional[str] = None
    state: MessageState = MessageState.PENDING
    error: Optional[str] = None
    duration: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        frozen = True

@dataclass(frozen=True)
class LLMConfig:
    model: str = "llama3.2:latest"
    temperature: float = 0.7
    base_url: str = "http://localhost:11434"
    stream: bool = False
    max_history: int = 8
    max_response_chars: int = 250
    system_prompt: str = """You are Sky, a friendly and helpful AI assistant. You are having a natural conversation.

Guidelines:
- Keep responses under 250 characters
- Use conversational, natural language
- Be concise but warm and helpful
- Ask follow-up questions when needed
- Speak as if you're having a phone conversation"""

class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = ChatOllama(
            base_url=self.config.base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            streaming=self.config.stream
        )
        
        self.memory = MemorySaver()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.current_thread_id = None

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self._process_message)
        return workflow

    def _truncate_response(self, response: str) -> str:
        if not response or len(response) <= self.config.max_response_chars:
            return response

        truncated = response[:self.config.max_response_chars]
        
        sentence_end = re.search(r'^(.*?[.!?])', truncated)
        if sentence_end:
            return sentence_end.group(1).strip()
        
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space].strip()
        
        return truncated.strip()

    def _process_message(self, state: MessagesState) -> Dict[str, Any]:
        selected_messages = trim_messages(
            state["messages"],
            token_counter=len,
            max_tokens=self.config.max_history,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False
        )
        
        if not any(isinstance(msg, SystemMessage) for msg in selected_messages):
            selected_messages.insert(0, SystemMessage(content=self.config.system_prompt))
        
        response = self.llm.invoke(selected_messages)
        
        if hasattr(response, 'content'):
            response.content = self._truncate_response(response.content)
        
        return {"messages": response}

    def generate(self, query: str) -> Message:
        try:
            if not self.current_thread_id:
                self.current_thread_id = uuid.uuid4()
                
            message = Message(query=query)
            start_time = datetime.utcnow()
            
            config = {"configurable": {"thread_id": self.current_thread_id}}
            input_message = HumanMessage(content=query)
            
            final_response = None
            for event in self.app.stream(
                {"messages": [input_message]}, 
                config, 
                stream_mode="values"
            ):
                final_response = event["messages"][-1].content
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return message.copy(
                update={
                    "response": final_response,
                    "state": MessageState.COMPLETE,
                    "duration": duration
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return message.copy(
                update={
                    "error": str(e),
                    "state": MessageState.ERROR,
                    "response": "I'm sorry, I didn't catch that. Could you please repeat?"
                }
            )

    def clear_memory(self) -> None:
        self.current_thread_id = None

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        if not self.current_thread_id:
            return []
        
        config = {"configurable": {"thread_id": self.current_thread_id}}
        try:
            state = self.memory.restore(config)
            if state and "messages" in state:
                return [
                    {
                        "query": msg.content if isinstance(msg, HumanMessage) else None,
                        "response": msg.content if not isinstance(msg, HumanMessage) else None,
                        "timestamp": datetime.utcnow()
                    }
                    for msg in state["messages"]
                    if not isinstance(msg, SystemMessage)
                ]
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
        
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass