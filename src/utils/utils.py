import base64
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests
import json
import glob
import datetime
import logging

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import gradio as gr

from .llm import DeepSeekR1ChatOpenAI, DeepSeekR1ChatOllama

# Get logger
logger = logging.getLogger(__name__)

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot"
}

def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider not in ["ollama"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            handle_api_key_error(provider, env_var)
        kwargs["api_key"] = api_key

    if provider == "anthropic":
        if not kwargs.get("base_url", ""):
            base_url = "https://api.anthropic.com"
        else:
            base_url = kwargs.get("base_url")

        return ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == 'mistral':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
        else:
            base_url = kwargs.get("base_url")
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("MISTRAL_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "deepseek":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("DEEPSEEK_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")

        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=api_key,
        )
    elif provider == "ollama":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        else:
            base_url = kwargs.get("base_url")

        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        api_version = kwargs.get("api_version", "") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    elif provider == "alibaba":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "moonshot":
        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("MOONSHOT_ENDPOINT"),
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Predefined model names for common providers
model_names = {
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-pro-exp-02-05"],
    "ollama": ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:14b", "qwen2.5-coder:32b", "llama2:7b", "deepseek-r1:14b", "deepseek-r1:32b"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"],
    "alibaba": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"],
    "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
}

# Callback to update the model name dropdown based on the selected provider
def update_model_dropdown(llm_provider, api_key=None, base_url=None):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use API keys from .env if not provided
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

    # Use predefined models for the selected provider
    if llm_provider in model_names:
        return gr.Dropdown(choices=model_names[llm_provider], value=model_names[llm_provider][0], interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)

def handle_api_key_error(provider: str, env_var: str):
    """
    Handles the missing API key error by raising a gr.Error with a clear message.
    """
    provider_display = PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
    raise gr.Error(
        f"💥 {provider_display} API key not found! 🔑 Please set the "
        f"`{env_var}` environment variable or provide it in the UI."
    )

def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")
            
    return latest_files
async def capture_screenshot(browser_context):
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    playwright_browser = browser_context.browser.playwright_browser  # Ensure this is correct.

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(
            type='jpeg',
            quality=75,
            scale="css"
        )
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception as e:
        return None

def save_task(task_name, task_description, additional_info=""):
    """
    Save a task to a JSON file for later use.
    
    Args:
        task_name (str): Name of the task to save
        task_description (str): The task description
        additional_info (str): Additional information for the task
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create tasks directory if it doesn't exist
        tasks_dir = os.path.join(os.getcwd(), "saved_tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        # Create the task data
        task_data = {
            "name": task_name,
            "description": task_description,
            "additional_info": additional_info,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Save to file
        filename = os.path.join(tasks_dir, f"{task_name.replace(' ', '_')}.json")
        with open(filename, 'w') as f:
            json.dump(task_data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving task: {str(e)}")
        return False

def load_tasks():
    """
    Load all saved tasks from the tasks directory.
    
    Returns:
        list: List of task dictionaries
    """
    tasks = []
    tasks_dir = os.path.join(os.getcwd(), "saved_tasks")
    
    # Create directory if it doesn't exist
    if not os.path.exists(tasks_dir):
        os.makedirs(tasks_dir, exist_ok=True)
        return tasks
    
    try:
        # Get all JSON files in the directory
        task_files = glob.glob(os.path.join(tasks_dir, "*.json"))
        
        for file_path in task_files:
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    tasks.append(task_data)
            except Exception as e:
                logger.error(f"Error loading task file {file_path}: {str(e)}")
                continue
                
        # Sort tasks by creation date (newest first)
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks
    except Exception as e:
        logger.error(f"Error loading tasks: {str(e)}")
        return []

def delete_task(task_name):
    """
    Delete a saved task.
    
    Args:
        task_name (str): Name of the task to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        tasks_dir = os.path.join(os.getcwd(), "saved_tasks")
        filename = os.path.join(tasks_dir, f"{task_name.replace(' ', '_')}.json")
        
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        return False

def save_playlist(playlist_name, task_ids, description=""):
    """
    Save a playlist of tasks to a JSON file.
    
    Args:
        playlist_name (str): Name of the playlist
        task_ids (list): List of task IDs/names in the playlist
        description (str): Optional description of the playlist
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create playlists directory if it doesn't exist
        playlists_dir = os.path.join(os.getcwd(), "saved_playlists")
        os.makedirs(playlists_dir, exist_ok=True)
        
        # Create the playlist data
        playlist_data = {
            "name": playlist_name,
            "description": description,
            "tasks": task_ids,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Save to file
        filename = os.path.join(playlists_dir, f"{playlist_name.replace(' ', '_')}.json")
        with open(filename, 'w') as f:
            json.dump(playlist_data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving playlist: {str(e)}")
        return False

def load_playlists():
    """
    Load all saved playlists from the playlists directory.
    
    Returns:
        list: List of playlist dictionaries
    """
    playlists = []
    playlists_dir = os.path.join(os.getcwd(), "saved_playlists")
    
    # Create directory if it doesn't exist
    if not os.path.exists(playlists_dir):
        os.makedirs(playlists_dir, exist_ok=True)
        return playlists
    
    try:
        # Get all JSON files in the directory
        playlist_files = glob.glob(os.path.join(playlists_dir, "*.json"))
        
        for file_path in playlist_files:
            try:
                with open(file_path, 'r') as f:
                    playlist_data = json.load(f)
                    playlists.append(playlist_data)
            except Exception as e:
                logger.error(f"Error loading playlist file {file_path}: {str(e)}")
                continue
                
        # Sort playlists by creation date (newest first)
        playlists.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return playlists
    except Exception as e:
        logger.error(f"Error loading playlists: {str(e)}")
        return []

def delete_playlist(playlist_name):
    """
    Delete a saved playlist.
    
    Args:
        playlist_name (str): Name of the playlist to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        playlists_dir = os.path.join(os.getcwd(), "saved_playlists")
        filename = os.path.join(playlists_dir, f"{playlist_name.replace(' ', '_')}.json")
        
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting playlist: {str(e)}")
        return False

def update_playlist_order(playlist_name, new_task_order):
    """
    Update the order of tasks in a playlist.
    
    Args:
        playlist_name (str): Name of the playlist to update
        new_task_order (list): New order of task IDs
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        playlists_dir = os.path.join(os.getcwd(), "saved_playlists")
        filename = os.path.join(playlists_dir, f"{playlist_name.replace(' ', '_')}.json")
        
        if not os.path.exists(filename):
            return False
            
        # Load existing playlist
        with open(filename, 'r') as f:
            playlist_data = json.load(f)
            
        # Update task order and updated_at timestamp
        playlist_data["tasks"] = new_task_order
        playlist_data["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save updated playlist
        with open(filename, 'w') as f:
            json.dump(playlist_data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error updating playlist order: {str(e)}")
        return False
