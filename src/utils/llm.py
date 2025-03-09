from openai import OpenAI
import pdb
from langchain_openai import ChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field, PrivateAttr

import json
import logging

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

# Configure logging
logger = logging.getLogger(__name__)

class DeepSeekR1ChatOpenAI(ChatOpenAI):
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        ) 
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
class DeepSeekR1ChatOllama(ChatOllama):
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = await super().ainvoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = super().invoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

def parse_json_response(text: str) -> Dict[str, Any]:
    """Parses a text response into a JSON object after basic cleanup.
    
    This function removes markdown code blocks and trims whitespace before 
    attempting to parse the text as JSON. If parsing fails, it returns an empty dict.
    
    Args:
        text: The text response from the model
        
    Returns:
        Parsed JSON as a dictionary, or empty dict if parsing fails
    """
    # Clean up content (remove markdown code blocks if present)
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {}

class QWQChatOpenAI(BaseChatModel):
    """Custom LangChain chat model for QWQ-32B via OpenRouter.
    
    This class handles the specific requirements of QWQ-32B, including:
    - Mapping OpenRouter's 'reasoning' field to 'reasoning_content'
    - Including the 'include_reasoning' parameter in requests
    - Handling both sync and async requests with shared logic
    """
    
    model_name: str = "qwen/qwq-32b"
    temperature: float = 0.0
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    api_key: str = Field(default="")
    http_referer: str = Field(default="https://browser-use-webui.com")
    x_title: str = Field(default="Browser Use WebUI")
    
    # Use private attribute for the client
    _client: Any = PrivateAttr()
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the QWQChatOpenAI model.
        
        Args:
            model_name: Model name to use
            temperature: Sampling temperature
            base_url: OpenRouter API endpoint
            api_key: OpenRouter API key
            http_referer: HTTP referer for OpenRouter tracking
            x_title: Title for OpenRouter tracking
        """
        # Extract model parameters before calling super().__init__
        model = kwargs.pop("model", "qwen/qwq-32b")
        kwargs["model_name"] = model
        
        super().__init__(**kwargs)
        
        # Initialize the OpenAI client with OpenRouter base URL
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": self.http_referer,
                "X-Title": self.x_title
            }
        )
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "qwq-openrouter"
    
    def _process_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI API format.
        
        Args:
            messages: List of LangChain messages
            
        Returns:
            List of messages in OpenAI API format
        """
        message_history = []
        for message in messages:
            if isinstance(message, SystemMessage):
                message_history.append({"role": "system", "content": message.content})
            elif isinstance(message, AIMessage):
                message_history.append({"role": "assistant", "content": message.content})
            else:
                message_history.append({"role": "user", "content": message.content})
        return message_history
    
    def _create_request_params(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Create parameters for the OpenRouter API request.
        
        Args:
            messages: List of messages in OpenAI API format
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of request parameters
        """
        # OpenRouter-specific parameters need to be passed via extra_body
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "extra_body": {
                "include_reasoning": True  # OpenRouter-specific parameter for reasoning tokens
            }
        }
    
    def _process_response(self, response: Any) -> AIMessage:
        """Process the OpenRouter API response.
        
        Args:
            response: Response from the OpenRouter API
            
        Returns:
            AIMessage with content and reasoning_content
        """
        # Extract reasoning and content (OpenRouter uses "reasoning" not "reasoning_content")
        reasoning_content = None
        if hasattr(response.choices[0].message, 'reasoning'):
            reasoning_content = response.choices[0].message.reasoning
            logger.info("🤯 Start Deep Thinking: ")
            logger.info(reasoning_content)
            logger.info("🤯 End Deep Thinking")
            
        content = response.choices[0].message.content
        
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LLMResult:
        """Generate a chat completion asynchronously."""
        raise NotImplementedError("Async generation not implemented yet")
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LLMResult:
        """Generate a chat completion."""
        raise NotImplementedError("Direct generation not implemented yet")
    
    async def ainvoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Asynchronously invoke the model with a list of messages.
        
        Args:
            input: List of messages
            config: Configuration for the runnable
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            AIMessage with content and reasoning_content
        """
        message_history = self._process_messages(input)
        request_params = self._create_request_params(message_history, **kwargs)
        
        # Make the API call
        response = await self._client.chat.completions.create(**request_params)
        
        return self._process_response(response)
    
    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Synchronously invoke the model with a list of messages.
        
        Args:
            input: List of messages
            config: Configuration for the runnable
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            AIMessage with content and reasoning_content
        """
        message_history = self._process_messages(input)
        request_params = self._create_request_params(message_history, **kwargs)
        
        # Make the API call
        response = self._client.chat.completions.create(**request_params)
        
        return self._process_response(response)