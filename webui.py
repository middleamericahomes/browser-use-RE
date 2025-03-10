import pdb
import logging
import shutil
import json
import threading
import queue
import time
import os
import glob
import asyncio
import argparse
import sys
import socket
import re

from dotenv import load_dotenv

# Create a class to collect and manage extracted data
class ExtractedDataCollector:
    def __init__(self):
        self.data = []
        self.data_by_type = {}  # Organize data by type
        self.data_by_source = {}  # Organize data by source
        self.data_by_timestamp = {}  # Organize data by timestamp (hour)
        self.lock = threading.Lock()
        self.metadata = {
            "total_items": 0,
            "types": set(),
            "sources": set(),
            "first_timestamp": None,
            "last_timestamp": None,
            "webpage_count": 0  # Track number of webpages captured
        }
    
    def add_data(self, json_data):
        """Add extracted data to the collection with metadata."""
        try:
            # Parse the JSON if it's a string
            if isinstance(json_data, str):
                data_obj = json.loads(json_data)
            else:
                data_obj = json_data
            
            # Add timestamp if not present
            if "timestamp" not in data_obj:
                data_obj["timestamp"] = time.time()
            
            # Add source if not present
            if "source" not in data_obj:
                data_obj["source"] = "unknown"
                
            # Add type if not present
            if "type" not in data_obj:
                # Try to infer type from content
                if "message" in data_obj:
                    data_obj["type"] = "log"
                elif "html" in data_obj:
                    data_obj["type"] = "html"
                    # Track webpage count for HTML content
                    self.metadata["webpage_count"] += 1
                elif "text" in data_obj:
                    data_obj["type"] = "text"
                elif "url" in data_obj:
                    # Ensure URL-related content is properly typed
                    data_obj["type"] = "webpage"
                    # Track webpage count
                    self.metadata["webpage_count"] += 1
                else:
                    data_obj["type"] = "unknown"
            
            # Ensure full content preservation for webpages
            if "html_content" in data_obj or "page_content" in data_obj or "url" in data_obj:
                # Mark as webpage data to ensure it's preserved in exports
                if "type" not in data_obj or data_obj["type"] not in ["html", "webpage"]:
                    data_obj["type"] = "webpage"
                    # Track webpage count
                    self.metadata["webpage_count"] += 1
            
            with self.lock:
                # Add to main data list
                self.data.append(data_obj)
                
                # Organize by type
                data_type = data_obj.get("type", "unknown")
                if not isinstance(data_type, str):
                    data_type = str(data_type)
                
                if data_type not in self.data_by_type:
                    self.data_by_type[data_type] = []
                self.data_by_type[data_type].append(data_obj)
                
                # Organize by source
                source = data_obj.get("source", "unknown")
                if not isinstance(source, str):
                    source = str(source)
                
                if source not in self.data_by_source:
                    self.data_by_source[source] = []
                self.data_by_source[source].append(data_obj)
                
                # Organize by timestamp (hour)
                timestamp = data_obj.get("timestamp", time.time())
                hour = time.strftime("%Y-%m-%d %H", time.localtime(timestamp))
                if hour not in self.data_by_timestamp:
                    self.data_by_timestamp[hour] = []
                self.data_by_timestamp[hour].append(data_obj)
                
                # Update metadata
                self.metadata["total_items"] += 1
                self.metadata["types"].add(data_type)
                self.metadata["sources"].add(source)
                
                if self.metadata["first_timestamp"] is None or timestamp < self.metadata["first_timestamp"]:
                    self.metadata["first_timestamp"] = timestamp
                
                if self.metadata["last_timestamp"] is None or timestamp > self.metadata["last_timestamp"]:
                    self.metadata["last_timestamp"] = timestamp
            
            return True
        except Exception as e:
            logging.error(f"Error adding data to collector: {str(e)}")
            return False
    
    def get_data(self):
        """Get all collected data as a list."""
        with self.lock:
            return self.data.copy()
    
    def get_data_json(self):
        """Get all collected data as a JSON string."""
        with self.lock:
            return json.dumps(self.data, indent=2)
    
    def get_data_by_type(self, data_type):
        """
        Get data filtered by type.
        
        Args:
            data_type: Type of data to filter (str or list)
            
        Returns:
            List of data items matching the type
        """
        with self.lock:
            # Handle potential list input from Gradio
            if isinstance(data_type, list):
                if len(data_type) > 0:
                    data_type = data_type[0]
                else:
                    return self.data.copy()
            
            # Convert to string to ensure it can be used as a dict key
            data_type_str = str(data_type) if data_type is not None else "unknown"
            
            if data_type_str == "all":
                return self.data.copy()
                
            return self.data_by_type.get(data_type_str, []).copy()
    
    def get_data_by_source(self, source):
        """
        Get data filtered by source.
        
        Args:
            source: Source of data to filter (str or list)
            
        Returns:
            List of data items matching the source
        """
        with self.lock:
            # Handle potential list input from Gradio
            if isinstance(source, list):
                if len(source) > 0:
                    source = source[0]
                else:
                    return self.data.copy()
            
            # Convert to string to ensure it can be used as a dict key
            source_str = str(source) if source is not None else "unknown"
            
            if source_str == "all":
                return self.data.copy()
                
            return self.data_by_source.get(source_str, []).copy()
    
    def get_data_by_timeframe(self, start_time, end_time=None):
        """Get data within a specific timeframe."""
        if end_time is None:
            end_time = time.time()
            
        with self.lock:
            return [item for item in self.data if start_time <= item.get("timestamp", 0) <= end_time]
    
    def get_metadata(self):
        """Get metadata about the collected data."""
        with self.lock:
            # Convert set to list for JSON serialization
            metadata = self.metadata.copy()
            metadata["types"] = list(metadata["types"])
            metadata["sources"] = list(metadata["sources"])
            
            # Format timestamps
            if metadata["first_timestamp"]:
                metadata["first_timestamp_formatted"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(metadata["first_timestamp"])
                )
            
            if metadata["last_timestamp"]:
                metadata["last_timestamp_formatted"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(metadata["last_timestamp"])
                )
                
            return metadata
    
    def clear_data(self):
        """Clear all collected data."""
        with self.lock:
            self.data = []
            self.data_by_type = {}
            self.data_by_source = {}
            self.data_by_timestamp = {}
            self.metadata = {
                "total_items": 0,
                "types": set(),
                "sources": set(),
                "first_timestamp": None,
                "last_timestamp": None
            }

# Create a custom log handler to capture extracted data
class ExtractedDataLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.process_logs = []
        self.log_categories = {
            "agent_action": [],
            "agent_thought": [],
            "agent_memory": [],
            "agent_planning": [],
            "agent_evaluation": [],
            "agent_task_progress": [],
            "extracted_data": [],
            "error": [],
            "system": [],
            "other": []
        }
        self.max_logs = 5000  # Increased from 2000 to 5000 for more history
    
    def emit(self, record):
        log_entry = self.format(record)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        
        # Create structured log entry
        structured_entry = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": log_entry,
            "source_file": record.pathname,
            "line_number": record.lineno,
            "step_number": None,  # Will be populated for step messages
            "raw_record": {
                "name": record.name,
                "levelname": record.levelname,
                "levelno": record.levelno,
                "pathname": record.pathname,
                "lineno": record.lineno,
                "msg": str(record.msg),
                "args": str(record.args),
                "exc_info": record.exc_info,
                "created": record.created
            }
        }
        
        # Store in process logs
        self.process_logs.append(structured_entry)
        
        # Categorize the log entry with enhanced detection
        if "Extracted from page" in log_entry and "```json" in log_entry:
            category = "extracted_data"
            # Extract the JSON part
            try:
                json_start = log_entry.find("```json") + 7
                json_end = log_entry.rfind("```")
                json_str = log_entry[json_start:json_end].strip()
                
                # Parse the JSON and add to the global list
                data_obj = json.loads(json_str)
                extracted_data_collector.add_data(data_obj)
                
                # Store the extracted JSON in the structured entry
                structured_entry["extracted_json"] = data_obj
            except Exception as e:
                logger.error(f"Error parsing extracted JSON: {str(e)}")
        elif record.levelname == "ERROR":
            category = "error"
        # Enhanced categorization with more specific categories
        elif "📍 Step" in log_entry and record.name == "src.agent.custom_agent":
            category = "agent_action"
            # Extract step number
            try:
                step_match = re.search(r"Step (\d+)", log_entry)
                if step_match:
                    structured_entry["step_number"] = int(step_match.group(1))
            except:
                pass
        elif "🤯 Start Deep Thinking" in log_entry or "🤯 End Deep Thinking" in log_entry:
            category = "agent_thought"
            structured_entry["thought_type"] = "deep_thinking"
        elif "🤯 Start Planning Deep Thinking" in log_entry or "🤯 End Planning Deep Thinking" in log_entry:
            category = "agent_planning"
            structured_entry["thought_type"] = "planning_deep_thinking"
        elif "🧠 All Memory" in log_entry:
            category = "agent_memory"
            # Extract memory content
            try:
                memory_content = log_entry.split("🧠 All Memory:")[1].strip()
                structured_entry["memory_content"] = memory_content
            except:
                structured_entry["memory_content"] = log_entry
        elif "🧠 New Memory" in log_entry:
            category = "agent_memory"
            structured_entry["memory_type"] = "new"
            # Extract memory content
            try:
                memory_content = log_entry.split("🧠 New Memory:")[1].strip()
                structured_entry["memory_content"] = memory_content
            except:
                structured_entry["memory_content"] = log_entry
        elif "⏳ Task Progress" in log_entry:
            category = "agent_task_progress"
            # Extract progress content
            try:
                progress_content = log_entry.split("⏳ Task Progress:")[1].strip()
                structured_entry["progress_content"] = progress_content
            except:
                structured_entry["progress_content"] = log_entry
        elif "📋 Future Plans" in log_entry or "📋 Plans" in log_entry:
            category = "agent_planning"
            # Extract planning content
            try:
                if "📋 Future Plans" in log_entry:
                    planning_content = log_entry.split("📋 Future Plans:")[1].strip()
                else:
                    planning_content = log_entry.split("📋 Plans:")[1].strip()
                structured_entry["planning_content"] = planning_content
            except:
                structured_entry["planning_content"] = log_entry
        elif "✅ Eval" in log_entry or "❌ Eval" in log_entry or "🤷 Eval" in log_entry:
            category = "agent_evaluation"
            # Extract evaluation status
            structured_entry["evaluation_status"] = "success" if "✅" in log_entry else ("failure" if "❌" in log_entry else "unknown")
            # Extract evaluation content
            try:
                eval_content = log_entry.split("Eval:")[1].strip()
                structured_entry["evaluation_content"] = eval_content
            except:
                structured_entry["evaluation_content"] = log_entry
        elif "🤔 Thought" in log_entry:
            category = "agent_thought"
            # Extract thought content
            try:
                thought_content = log_entry.split("🤔 Thought:")[1].strip()
                structured_entry["thought_content"] = thought_content
            except:
                structured_entry["thought_content"] = log_entry
        elif "🎯 Summary" in log_entry:
            category = "agent_thought"
            structured_entry["thought_type"] = "summary"
            # Extract summary content
            try:
                summary_content = log_entry.split("🎯 Summary:")[1].strip()
                structured_entry["summary_content"] = summary_content
            except:
                structured_entry["summary_content"] = log_entry
        elif "🛠️  Action" in log_entry:
            category = "agent_action"
            # Extract action details
            try:
                action_match = re.search(r"Action (\d+)/(\d+)", log_entry)
                if action_match:
                    structured_entry["action_number"] = int(action_match.group(1))
                    structured_entry["total_actions"] = int(action_match.group(2))
                action_content = log_entry.split(": ", 1)[1] if ": " in log_entry else log_entry
                structured_entry["action_content"] = action_content
            except:
                structured_entry["action_content"] = log_entry
        elif record.name.startswith("src.") or record.name.startswith("browser_use."):
            category = "system"
        else:
            category = "other"
        
        # Add to the appropriate category
        structured_entry["category"] = category
        self.log_categories[category].append(structured_entry)
        
        # Keep only the last max_logs messages to prevent memory issues
        if len(self.process_logs) > self.max_logs:
            self.process_logs = self.process_logs[-self.max_logs:]
            
        # Also trim category logs
        for cat in self.log_categories:
            if len(self.log_categories[cat]) > self.max_logs // 2:
                self.log_categories[cat] = self.log_categories[cat][-(self.max_logs // 2):]
    
    def get_process_logs(self):
        """Get all process logs as a single string."""
        return "\n".join([f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}" for entry in self.process_logs])
    
    def get_logs_by_category(self, category):
        """
        Get logs for a specific category.
        
        Args:
            category: The category to filter by (string or list)
            
        Returns:
            String containing formatted log entries
        """
        try:
            # Handle potential list input
            if isinstance(category, list):
                if len(category) > 0:
                    category = category[0]
                else:
                    return self.get_process_logs()
            
            # Convert to string to ensure it can be used as a dict key
            category_str = str(category) if category is not None else "all"
            
            if category_str == "all":
                return self.get_process_logs()
                
            if category_str in self.log_categories:
                return "\n".join([
                    f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}" 
                    for entry in self.log_categories[category_str]
                ])
            return ""
        except Exception as e:
            logging.error(f"Error getting logs by category: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_logs_by_level(self, level):
        """
        Get logs filtered by log level.
        
        Args:
            level: The log level to filter by (string or list)
            
        Returns:
            String containing formatted log entries
        """
        try:
            # Handle potential list input
            if isinstance(level, list):
                if len(level) > 0:
                    level = level[0]
                else:
                    return self.get_process_logs()
            
            # Convert to string to ensure proper comparison
            level_str = str(level) if level is not None else "ALL"
            
            if level_str == "ALL":
                return self.get_process_logs()
                
            # Convert to uppercase for case-insensitive comparison
            level_upper = level_str.upper() if isinstance(level_str, str) else level_str
                
            filtered_logs = [
                entry for entry in self.process_logs 
                if entry['level'] == level_upper
            ]
            
            return "\n".join([
                f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}" 
                for entry in filtered_logs
            ])
        except Exception as e:
            logging.error(f"Error getting logs by level: {str(e)}")
            return f"Error: {str(e)}"
    
    def search_logs(self, search_term):
        """
        Search logs for a specific term.
        
        Args:
            search_term: The term to search for (string or list)
            
        Returns:
            String containing formatted log entries matching the search
        """
        try:
            # Handle potential list input
            if isinstance(search_term, list):
                if len(search_term) > 0:
                    search_term = search_term[0]
                else:
                    return self.get_process_logs()
                    
            # Convert to string to ensure proper searching
            search_str = str(search_term) if search_term is not None else ""
            
            if not search_str:
                return self.get_process_logs()
                
            # Convert to lowercase for case-insensitive search
            search_lower = search_str.lower() if isinstance(search_str, str) else search_str
                
            filtered_logs = [
                entry for entry in self.process_logs 
                if search_lower in str(entry.get('message', '')).lower()
            ]
            
            return "\n".join([
                f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}" 
                for entry in filtered_logs
            ])
        except Exception as e:
            logging.error(f"Error searching logs: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_stats(self):
        """Get statistics about the logs."""
        stats = {
            "total_logs": len(self.process_logs),
            "by_category": {cat: len(logs) for cat, logs in self.log_categories.items()},
            "by_level": {}
        }
        
        # Count logs by level
        for entry in self.process_logs:
            level = entry['level']
            if level not in stats["by_level"]:
                stats["by_level"][level] = 0
            stats["by_level"][level] += 1
            
        return stats
    
    def clear_logs(self):
        """Clear all process logs."""
        self.process_logs = []
        for category in self.log_categories:
            self.log_categories[category] = []

# Create a custom log handler to capture terminal output
class TerminalLogHandler(logging.Handler):
    def __init__(self, data_collector):
        super().__init__()
        self.data_collector = data_collector
    
    def emit(self, record):
        log_entry = self.format(record)
        # Add the log entry to the data collector
        self.data_collector.add_data({
            "type": "terminal_log",
            "message": log_entry,
            "level": record.levelname,
            "timestamp": record.created
        })
        
        # Also add to agent tracker for inclusion in JSON export
        try:
            tracker = get_agent_tracker()
            if tracker:
                tracker.add_terminal_log(log_entry)
        except Exception as e:
            # Don't let errors in tracking disrupt logging
            pass

# Create global instances for data collection and logging
extracted_data_collector = ExtractedDataCollector()
data_log_handler = ExtractedDataLogHandler()

# Function to get or create the global agent tracker (prevents order dependency)
_agent_tracker_instance = None
def get_agent_tracker():
    """Get or create the global AgentTracker instance"""
    global _agent_tracker_instance
    if _agent_tracker_instance is None:
        # Use the AgentTracker class defined in this file
        _agent_tracker_instance = AgentTracker()
    return _agent_tracker_instance

# Custom log interceptor to feed data to the agent tracker
class AgentTrackerLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        
    def emit(self, record):
        try:
            log_entry = self.format(record)
            
            # Only process agent-related logs
            if record.name != 'src.agent.custom_agent':
                return
                
            # Extract action data
            if "🛠️  Action" in log_entry:
                try:
                    # Parse action details from log
                    action_parts = log_entry.split("🛠️  Action")[1].strip()
                    action_num = action_parts.split("/")[0].strip()
                    action_content = action_parts.split(":", 1)[1].strip() if ":" in action_parts else action_parts
                    
                    # Track the action
                    get_agent_tracker().track_action({
                        "action_type": "browser_action",
                        "action_number": action_num,
                        "content": action_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking action: {str(e)}")
            
            # Extract thought data
            elif "🤔 Thought" in log_entry:
                try:
                    thought_content = log_entry.split("🤔 Thought:")[1].strip()
                    get_agent_tracker().track_thought({
                        "thought_type": "reasoning",
                        "content": thought_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking thought: {str(e)}")
            
            # Extract memory data
            elif "🧠 New Memory" in log_entry:
                try:
                    memory_content = log_entry.split("🧠 New Memory:")[1].strip()
                    get_agent_tracker().track_memory({
                        "memory_type": "new",
                        "content": memory_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking memory: {str(e)}")
            
            # Extract evaluation data
            elif "Eval:" in log_entry:
                try:
                    success = "✅" in log_entry
                    failed = "❌" in log_entry
                    eval_content = log_entry.split("Eval:")[1].strip()
                    
                    get_agent_tracker().track_evaluation({
                        "success": success,
                        "failed": failed,
                        "content": eval_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking evaluation: {str(e)}")
            
            # Extract planning data
            elif "📋 Future Plans" in log_entry:
                try:
                    plan_content = log_entry.split("📋 Future Plans:")[1].strip()
                    get_agent_tracker().track_plan({
                        "plan_type": "future",
                        "content": plan_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking plan: {str(e)}")
            
            # Extract step information to track progress
            elif "📍 Step" in log_entry:
                try:
                    step_match = re.search(r"Step (\d+)", log_entry)
                    if step_match:
                        step_num = int(step_match.group(1))
                        get_agent_tracker().track_action({
                            "action_type": "step_transition",
                            "step": step_num,
                            "raw_log": log_entry
                        })
                except Exception as e:
                    logging.error(f"Error tracking step: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in AgentTrackerLogHandler: {str(e)}")

# Create and add the agent tracker handler
agent_tracker_handler = AgentTrackerLogHandler()
agent_tracker_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('src.agent.custom_agent').addHandler(agent_tracker_handler)

# Create and add the agent tracker handler
agent_tracker_handler = AgentTrackerLogHandler()

# Add the data handler to relevant loggers
controller_logger = logging.getLogger('controller')
controller_logger.addHandler(data_log_handler)

# Set up the terminal log handler
terminal_logger = logging.getLogger()
terminal_logger.setLevel(logging.INFO)
terminal_log_handler = TerminalLogHandler(extracted_data_collector)
terminal_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
terminal_logger.addHandler(terminal_log_handler)

# Add a stream handler to output logs to the terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler.setLevel(logging.INFO)
terminal_logger.addHandler(stream_handler)

# Function to log messages to both terminal and data collector
def log_message(message, level="INFO"):
    """Log a message to both terminal and data collector."""
    if level.upper() == "INFO":
        logging.info(message)
    elif level.upper() == "WARNING":
        logging.warning(message)
    elif level.upper() == "ERROR":
        logging.error(message)
    elif level.upper() == "DEBUG":
        logging.debug(message)
    else:
        logging.info(message)
    
    # Also add directly to data collector for immediate display
    extracted_data_collector.add_data({
        "type": "direct_log",
        "message": message,
        "level": level,
        "timestamp": time.time()
    })

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

# Add the handler to the root logger and other relevant loggers
root_logger = logging.getLogger()
root_logger.addHandler(data_log_handler)

# Also add to specific loggers that might be used by the agent
agent_logger = logging.getLogger('agent')
agent_logger.addHandler(data_log_handler)
browser_logger = logging.getLogger('browser_use')
browser_logger.addHandler(data_log_handler)
controller_logger = logging.getLogger('controller')
controller_logger.addHandler(data_log_handler)
src_agent_logger = logging.getLogger('src.agent.custom_agent')
src_agent_logger.addHandler(data_log_handler)

import gradio as gr
from gradio import themes

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot, save_task, load_tasks, delete_task
from src.utils.playlist_utils import (
    load_saved_playlists_for_ui,
    load_selected_playlist_for_ui,
    save_current_playlist_for_ui,
    delete_selected_playlist_for_ui,
    add_task_to_playlist_for_ui,
    remove_task_from_playlist_for_ui,
    move_task_up_for_ui,
    move_task_down_for_ui,
    update_available_tasks_for_ui,
    play_playlist_for_ui
)

# Ensure default_config includes 'add_infos'
def get_config_with_add_infos():
    """Get the default config and ensure it has the 'add_infos' key."""
    config = default_config()
    if 'add_infos' not in config:
        config['add_infos'] = ""  # Add with empty string as default value
    return config

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

# Global variable to store extracted data
extracted_data = []
extracted_data_lock = threading.Lock()

def add_extracted_data(data_obj):
    """Add extracted data to the collector."""
    global extracted_data_collector
    return extracted_data_collector.add_data(data_obj)

def get_extracted_data():
    """Get all collected data."""
    global extracted_data_collector
    return extracted_data_collector.get_data()

def clear_extracted_data():
    """Clear all collected data."""
    global extracted_data_collector
    extracted_data_collector.clear_data()
    return []

def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text
        
    import re
    
    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)
    
    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)
        
    return result

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser, _global_agent

    try:
        # Request stop
        if _global_agent is not None:
                _global_agent.stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )
        
async def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (                                   # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    """Run the browser agent with the specified parameters."""
    global _global_browser, _global_browser_context, _global_agent_state, _global_agent
    
    # Log the start of the agent
    log_message(f"Starting {agent_type} agent with task: {task}")
    
    # Check if an agent is already running
    if _global_agent is not None:
        log_message("An agent is already running. Please stop it first.", "ERROR")
        return "An agent is already running. Please stop it first."
    
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',                                         # final_result
            errors,                                     # errors
            '',                                         # model_actions
            '',                                         # model_thoughts
            None,                                       # latest_video
            None,                                       # history_file
            None,                                       # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent
        
        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None
            
        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent

        # Clear any previous stop request
        if _global_agent_state is not None:
            _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )
            
        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        # Create a sanitized task name for the filename (remove special characters)
        import re
        import datetime
        
        # Get the first 50 characters of the task and sanitize it for filename use
        task_short = task[:50] if task else "unknown_task"
        task_sanitized = re.sub(r'[^\w\s-]', '', task_short).strip().replace(' ', '_')
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filename with task name, timestamp, and agent ID
        descriptive_filename = f"{task_sanitized}_{timestamp}_{_global_agent.agent_id}.json"
        
        # Save with original UUID filename for compatibility
        original_history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(original_history_file)
        
        # Also save with descriptive filename
        descriptive_history_file = os.path.join(save_agent_history_path, descriptive_filename)
        _global_agent.save_history(descriptive_history_file)
        
        # Auto-export the history file to user's downloads
        auto_export_history(descriptive_history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)        

        # Return the descriptive filename for display in the UI
        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), descriptive_history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

# Add a function to automatically export the agent history
def auto_export_history(history_file_path):
    """
    Automatically export the agent history file to the user's downloads directory.
    
    Args:
        history_file_path (str): Path to the agent history file
    
    Returns:
        str: Path to the exported file or error message
    """
    if not history_file_path or not os.path.exists(history_file_path):
        logging.warning("No agent history file available to auto-export.")
        return None
    
    try:
        # Get the filename from the path
        filename = os.path.basename(history_file_path)
        
        # Create a user downloads directory if it doesn't exist
        user_downloads = os.path.expanduser("~/Downloads/browser-use-exports")
        os.makedirs(user_downloads, exist_ok=True)
        
        # Copy the file to the user's downloads directory
        export_path = os.path.join(user_downloads, filename)
        shutil.copy2(history_file_path, export_path)
        
        logging.info(f"Agent history auto-exported to: {export_path}")
        return export_path
    except Exception as e:
        logging.error(f"Error auto-exporting agent history: {str(e)}")
        return None

async def run_with_stream(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    
    # Log the start of the streaming process
    log_message(f"Starting agent with streaming output for task: {task}")
    
    # Clear the logs at the start of a new run
    data_log_handler.clear_logs()
    extracted_data_collector.clear_data()
    
    # Create a function to update the process log
    def update_logs():
        process_logs = data_log_handler.get_process_logs()
        return process_logs
    
    if not headless:
        log_message("Running in non-headless mode")
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        # Add HTML content at the start of the result array
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        
        # Get the extracted data
        extracted_data = extracted_data_collector.get_data()
        
        # Get the process logs
        process_logs = update_logs()
        
        # Insert the extracted data into the result
        result_list = list(result)
        result_list.insert(1, extracted_data)  # Insert extracted data after html_content
        
        log_message("Agent completed execution in non-headless mode")
        
        # Add process logs to the result
        yield [html_content] + result_list + [process_logs]
    else:
        log_message("Running in headless mode")
        try:
            _global_agent_state.clear_stop()
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method
                )
            )
            # Wait for the agent to complete
            result = await agent_task
            
            # Get the process logs
            process_logs = update_logs()
            
            log_message("Agent completed execution in headless mode")
            yield result + [process_logs]
        except Exception as e:
            log_message(f"Error during headless execution: {str(e)}")
            yield (
                '',                                         # final_result
                f"Error during headless execution: {str(e)}",  # errors
                '',                                         # model_actions
                '',                                         # model_thoughts
                None,                                       # latest_video
                None,                                       # history_file
                None,                                       # trace_file
                gr.update(value="Stop", interactive=True),      # Re-enable stop button
                gr.update(interactive=True),                   # Re-enable run button
                ''                                          # process_logs
            )

# Define the theme map globally
theme_map = {
    "Default": Default,
    "Soft": Soft,
    "Monochrome": Monochrome,
    "Glass": Glass,
    "Origin": Origin,
    "Citrus": Citrus,
    "Ocean": Ocean,
    "Base": Base
}

async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None
        
async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser, headless):
    from src.utils.deep_research import deep_research
    global _global_agent_state

    # Clear any previous stop request
    _global_agent_state.clear_stop()
    
    llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                        max_search_iterations=max_search_iteration_input,
                                                        max_query_num=max_query_per_iter_input,
                                                        use_vision=use_vision,
                                                        headless=headless,
                                                        use_own_browser=use_own_browser
                                                        )
    
    return markdown_content, file_path, gr.update(value="Stop", interactive=True),  gr.update(interactive=True) 
    

def create_ui(config, theme_name="Ocean"):
    # Get the theme class
    theme_instance = theme_name
    
    css = """
    /* Custom CSS */
    .gradio-container {
        max-width: 1400px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Make JSON component wider */
    .json-component {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Improve JSON formatting */
    .json-component pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_instance, css=css
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value=config['agent_type'],
                        info="Select the type of agent to use",
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=config['max_steps'],
                            step=1,
                            label="Max Run Steps",
                            info="Maximum number of steps the agent will take",
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config['max_actions_per_step'],
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent will take per step",
                        )
                    with gr.Column():
                        use_vision = gr.Checkbox(
                            label="Use Vision",
                            value=config['use_vision'],
                            info="Enable visual processing capabilities",
                        )
                        tool_calling_method = gr.Dropdown(
                            label="Tool Calling Method",
                            value=config['tool_calling_method'],
                            interactive=True,
                            allow_custom_value=True,  # Allow users to input custom model names
                            choices=["auto", "json_schema", "function_calling"],
                            info="Tool Calls Funtion Name",
                            visible=False
                        )

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=[provider for provider,model in utils.model_names.items()],
                        label="LLM Provider",
                        value=config['llm_provider'],
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=utils.model_names['openai'],
                        value=config['llm_model_name'],
                        interactive=True,
                        allow_custom_value=True,  # Allow users to input custom model names
                        info="Select a model from the dropdown or type a custom model name"
                    )
                    llm_num_ctx = gr.Slider(
                        minimum=2**8,
                        maximum=2**16,
                        value=config['llm_num_ctx'],
                        step=1,
                        label="Max Context Length",
                        info="Controls max context length model needs to handle (less = faster)",
                        visible=config['llm_provider'] == "ollama"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config['llm_temperature'],
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value=config['llm_base_url'],
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=config['llm_api_key'],
                            info="Your API key (leave blank to use .env)"
                        )

            # Change event to update context length slider
            def update_llm_num_ctx_visibility(llm_provider):
                return gr.update(visible=llm_provider == "ollama")

            # Bind the change event of llm_provider to update the visibility of context length slider
            llm_provider.change(
                fn=update_llm_num_ctx_visibility,
                inputs=llm_provider,
                outputs=llm_num_ctx
            )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=config['use_own_browser'],
                            info="Use your existing browser instance",
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=config['keep_browser_open'],
                            info="Keep Browser Open between Tasks",
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=config['headless'],
                            info="Run browser without GUI",
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=config['disable_security'],
                            info="Disable browser security features",
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=config['enable_recording'],
                            info="Enable saving browser recordings",
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=config['window_w'],
                            info="Browser window width",
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=config['window_h'],
                            info="Browser window height",
                        )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value=config['save_recording_path'],
                        info="Path to save browser recordings",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value=config['save_trace_path'],
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value=config['save_agent_history_path'],
                        info="Specify the directory where agent history should be saved.",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Run Agent", id=4):
                # Add task template dropdown at the top
                with gr.Row():
                    task_template_dropdown = gr.Dropdown(
                        label="📋 Task Templates",
                            choices=[],
                        value=None,
                        info="Select a saved task template",
                            interactive=True,
                        allow_custom_value=False,
                    )
                    refresh_templates_button = gr.Button("🔄 Refresh", scale=1)
                    save_task_button = gr.Button("💾 Save Task", variant="secondary", scale=1)
                
                # Add feedback message
                task_feedback = gr.Textbox(label="", visible=False)
                
                # Add task name input field
                task_name_input = gr.Textbox(
                    label="✏️ Task Name",
                    placeholder="Enter a name for this task",
                    info="A short descriptive name for the task"
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                                task = gr.Textbox(
                            label="Task",
                            lines=5,
                            value=config['task'],
                                    info="Describe the task for the agent to perform",
                        )
                    add_infos = gr.Textbox(
                            label="Additional Information",
                            lines=3,
                            value=config['add_infos'],
                            info="Optional: Provide additional context or instructions",
                        )
                        
                        # Add live log for extracted data directly on the Run Agent tab
                    with gr.Accordion("📋 Live Data Log", open=True):
                            extracted_data_log = gr.JSON(
                                label="Extracted Data",
                                show_label=True,
                                elem_id="extracted_data_log",
                                value=extracted_data_collector.get_data(),  # Initialize with current data
                                height=500,  # Make the box taller
                                container=True,  # Add container styling
                            )
                            
                            # Add custom CSS to make the JSON component wider
                            gr.HTML("""
                            <style>
                            #extracted_data_log {
                                width: 100% !important;
                                max-width: 100% !important;
                            }
                            .json-component {
                                width: 100% !important;
                                max-width: 100% !important;
                            }
                            </style>
                            """)
                            
                            # Add export status message
                            export_status = gr.Textbox(
                                label="Export Status",
                                visible=True,
                                interactive=False
                            )
                            
                            with gr.Row():
                                export_data_button = gr.Button("📤 Export Data", variant="primary")
                                clear_data_button = gr.Button("🧹 Clear Data", variant="secondary")
                        
                    with gr.Row():
                            run_button = gr.Button("▶️ Run", variant="primary", scale=2)
                            stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                    
                    with gr.Column(scale=3):
                        stream_output = gr.HTML(
                            label="Browser View",
                            value="<h1 style='width:80vw; height:60vh'>Browser view will appear here when running</h1>",
                        )
            
            with gr.TabItem("🎮 Task Playlists", id=5):
                # Playlist management
                with gr.Row():
                    with gr.Column(scale=3):
                        playlist_name_input = gr.Textbox(
                            label="Playlist Name",
                            placeholder="Enter a name for your playlist",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        save_playlist_button = gr.Button("💾 Save Playlist")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        playlist_description_input = gr.Textbox(
                            label="Description",
                            placeholder="Enter a description for your playlist",
                            interactive=True
                        )
                
                # Initialize playlists
                playlists = load_saved_playlists_for_ui()
                logger.info(f"Initialized playlist dropdown with {len(playlists)} playlists")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        playlist_dropdown = gr.Dropdown(
                            label="Saved Playlists",
                            choices=playlists,
                            value=None,  # Start with no selection
                            info="Select a playlist to load",
                            interactive=True,
                            allow_custom_value=False,
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            refresh_playlists_button = gr.Button("🔄 Refresh")
                            load_playlist_button = gr.Button("📂 Load")
                    with gr.Column(scale=1):
                        delete_playlist_button = gr.Button("🗑️ Delete")
                
                # Playlist status message
                playlist_feedback = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=True
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Use regular Dataframe instead of CustomDataFrameComponent
                        playlist_tasks = gr.Dataframe(
                            headers=["Task Name", "Description"],
                            datatype=["str", "str"],
                            row_count=10,
                            col_count=(2, "fixed"),
                            interactive=False,
                            wrap=True
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            move_up_button = gr.Button("⬆️ Move Up")
                        with gr.Row():
                            move_down_button = gr.Button("⬇️ Move Down")
                        with gr.Row():
                            remove_task_button = gr.Button("❌ Remove")
                
                # Available tasks to add to playlist
                with gr.Row():
                    with gr.Column(scale=3):
                        available_tasks_dropdown = gr.Dropdown(
                            label="Available Tasks",
                            choices=[],
                            value=None,  # Start with no selection
                            info="Select a task to add to the playlist",
                            interactive=True,
                            allow_custom_value=False,
                        )
                    with gr.Column(scale=1):
                        refresh_tasks_button = gr.Button("🔄 Refresh Tasks")
                        add_to_playlist_button = gr.Button("➕ Add to Playlist")
                
                # Playlist controls
                with gr.Row():
                    play_playlist_button = gr.Button("▶️ Play Playlist", variant="primary", scale=2)
                    stop_playlist_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                
                # Current task display
                current_task_display = gr.Textbox(
                    label="Current Task",
                    interactive=False,
                    visible=True
                )
                
                # Connect event handlers
                refresh_playlists_button.click(
                    fn=load_saved_playlists_for_ui,
                    inputs=[],
                    outputs=[playlist_dropdown]
                )
                
                load_playlist_button.click(
                    fn=load_selected_playlist_for_ui,
                    inputs=[playlist_dropdown],
                    outputs=[playlist_tasks, playlist_name_input, playlist_description_input, playlist_feedback]
                )
                
                save_playlist_button.click(
                    fn=save_current_playlist_for_ui,
                    inputs=[playlist_name_input, playlist_description_input, playlist_tasks],
                    outputs=[playlist_dropdown, playlist_feedback, playlist_tasks]
                )
                
                delete_playlist_button.click(
                    fn=delete_selected_playlist_for_ui,
                    inputs=[playlist_dropdown],
                    outputs=[playlist_dropdown, playlist_feedback]
                )
                
                # Initialize available tasks dropdown
                refresh_tasks_button.click(
                    fn=update_available_tasks_for_ui,
                    inputs=[],
                    outputs=[available_tasks_dropdown]
                )
                
                # Add task to playlist
                add_to_playlist_button.click(
                    fn=add_task_to_playlist_for_ui,
                    inputs=[available_tasks_dropdown, playlist_tasks],
                    outputs=[playlist_tasks, playlist_feedback]
                )
                
                # Remove task from playlist
                remove_task_button.click(
                    fn=remove_task_from_playlist_for_ui,
                    inputs=[gr.State(0), playlist_tasks],  # Default to first task
                    outputs=[playlist_tasks, playlist_feedback]
                )
                
                # Move task up
                move_up_button.click(
                    fn=move_task_up_for_ui,
                    inputs=[gr.State(0), playlist_tasks],  # Default to first task
                    outputs=[playlist_tasks, playlist_feedback]
                )
                
                # Move task down
                move_down_button.click(
                    fn=move_task_down_for_ui,
                    inputs=[gr.State(0), playlist_tasks],  # Default to first task
                    outputs=[playlist_tasks, playlist_feedback]
                )
                
                # Play playlist
                play_playlist_button.click(
                    fn=play_playlist_for_ui,
                    inputs=[
                        playlist_tasks,
                        agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                        save_recording_path, save_agent_history_path, save_trace_path,
                        enable_recording, max_steps, use_vision, max_actions_per_step, tool_calling_method
                    ],
                    outputs=[
                        current_task_display,
                        play_playlist_button,
                        stop_playlist_button,
                        playlist_feedback
                    ]
                )
                
                # Stop playlist
                stop_playlist_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[playlist_feedback, stop_playlist_button, play_playlist_button]
                )
            
            with gr.TabItem("🧐 Deep Research", id=6):
                research_task_input = gr.Textbox(label="Research Task", lines=5, value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.")
                with gr.Row():
                    max_search_iteration_input = gr.Number(label="Max Search Iteration", value=3, precision=0) # precision=0 确保是整数
                    max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=1, precision=0) # precision=0 确保是整数
                with gr.Row():
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")


            with gr.TabItem("📊 Results", id=7):
                with gr.Group():
                    # Add tabs for different types of logs
                    with gr.Tabs():
                        with gr.TabItem("📋 Extracted Data Log"):
                            # Add data filtering controls
                            with gr.Row():
                                with gr.Column(scale=1):
                                    # Common data types that should always be available
                                    default_data_types = ["all", "terminal_log", "agent_action", "agent_thought", 
                                                         "extracted_data", "webpage", "html", "error", "system", "other", "unknown"]
                                    data_type_dropdown = gr.Dropdown(
                                        choices=default_data_types,
                                        value="all",
                                        label="Data Type",
                                        interactive=True
                                    )
                                with gr.Column(scale=1):
                                    # Common data sources that should always be available
                                    default_data_sources = ["all", "terminal_log", "agent_action", "agent_thought", 
                                                          "extracted_data", "error", "system", "other", "unknown"]
                                    data_source_dropdown = gr.Dropdown(
                                        choices=default_data_sources,
                                        value="all",
                                        label="Data Source",
                                        interactive=True
                                    )
                                with gr.Column(scale=2):
                                    data_search_input = gr.Textbox(
                                        label="Search Data",
                                        placeholder="Enter search term...",
                                        interactive=True
                                    )
                            
                            # Add data statistics display
                            with gr.Row():
                                data_stats_json = gr.JSON(label="Data Statistics", value={})
                            
                            # Extracted data display
                            extracted_data_log = gr.JSON(
                                label="Extracted Data",
                                value=extracted_data_collector.get_data(),
                                height=500,
                                elem_id="extracted_data_log"
                            )
                            
                            # Add custom CSS to make the JSON component wider
                            gr.HTML("""
                            <style>
                            #extracted_data_log {
                                width: 100% !important;
                            }
                            .json-component {
                                width: 100% !important;
                                max-width: 100% !important;
                            }
                            .json-component pre {
                                font-size: 14px !important;
                                line-height: 1.5 !important;
                                white-space: pre-wrap !important;
                                word-break: break-word !important;
                            }
                            </style>
                            """)
                            
                            # Buttons for data management
                            with gr.Row():
                                update_extracted_data_button = gr.Button("🔄 Refresh Data", variant="primary")
                                export_data_button = gr.Button("💾 Export Data")
                                clear_data_button = gr.Button("🗑️ Clear Data")
                        
                        with gr.TabItem("🔄 Process Log"):
                            # Add log filtering controls
                            with gr.Row():
                                with gr.Column(scale=1):
                                    log_category_dropdown = gr.Dropdown(
                                        choices=["all", "agent_action", "agent_thought", "extracted_data", "error", "system", "other", "terminal_log", "unknown"],
                                        value="all",
                                        label="Log Category",
                                        interactive=True
                                    )
                                with gr.Column(scale=1):
                                    log_level_dropdown = gr.Dropdown(
                                        choices=["ALL", "INFO", "WARNING", "ERROR", "DEBUG"],
                                        value="ALL",
                                        label="Log Level",
                                        interactive=True
                                    )
                                with gr.Column(scale=2):
                                    log_search_input = gr.Textbox(
                                        label="Search Logs",
                                        placeholder="Enter search term...",
                                        interactive=True
                                    )
                            
                            # Add log statistics display
                            with gr.Row():
                                log_stats_json = gr.JSON(label="Log Statistics", value={})
                            
                            # Process log display
                            process_log_data = gr.Textbox(
                                label="Process Log",
                                value="",
                                lines=25,
                                max_lines=50,
                                interactive=False,
                                elem_id="process_log_data"
                            )
                            
                            # Add custom CSS to make the process log look like a terminal
                            gr.HTML("""
                            <style>
                            #process_log_data textarea {
                                font-family: 'Courier New', monospace !important;
                                background-color: #1e1e1e !important;
                                color: #f0f0f0 !important;
                                padding: 10px !important;
                                border-radius: 5px !important;
                                border: 1px solid #444 !important;
                                min-height: 600px !important;
                                height: 600px !important;
                                overflow-y: auto !important;
                                white-space: pre !important;
                                line-height: 1.4 !important;
                                font-size: 14px !important;
                            }
                            </style>
                            """)
                            
                            # Buttons for log management
                            with gr.Row():
                                update_process_log_button = gr.Button("🔄 Refresh Log", variant="primary")
                                export_process_log_button = gr.Button("💾 Export Log")
                                clear_process_log_button = gr.Button("🗑️ Clear Log")

                    recording_display = gr.Video(label="Latest Recording")

                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result", lines=3, show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors", lines=3, show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions", lines=3, show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts", lines=3, show_label=True
                            )

                    trace_file = gr.File(label="Trace File")

                    agent_history_file = gr.File(label="Agent History")
                    
                    # Add export button for agent history
                    with gr.Row():
                        export_history_button = gr.Button("📤 Export Agent History", variant="secondary")

            with gr.TabItem("📁 Configuration", id=9):
                with gr.Group():
                    config_file_input = gr.File(
                        label="Load Config File",
                        file_types=[".pkl"],
                        interactive=True
                    )

                    load_config_button = gr.Button("Load Existing Config From File", variant="primary")
                    save_config_button = gr.Button("Save Current Config", variant="primary")

                    config_status = gr.Textbox(
                        label="Status",
                        lines=2,
                        interactive=False
                    )

                load_config_button.click(
                    fn=update_ui_from_config,
                    inputs=[config_file_input],
                    outputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, enable_recording,
                        window_w, window_h, save_recording_path, save_trace_path, save_agent_history_path,
                        task, config_status
                    ]
                )

                save_config_button.click(
                    fn=save_current_config,
                    inputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                        save_agent_history_path, task,
                    ],  
                    outputs=[config_status]
                )


        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

        # Function to update task templates dropdown
        def update_task_templates_dropdown():
            tasks = load_tasks()
            return gr.Dropdown(choices=[task["name"] for task in tasks])
        
        # Function to load selected task template
        def load_task_template(template_name):
            if not template_name:
                return "", "", ""
            
            tasks = load_tasks()
            for task in tasks:
                if task["name"] == template_name:
                    return task["name"], task.get("description", ""), task.get("additional_info", "")
            
            return "", "", ""
        
        # Function to save current task
        def save_current_task(task_name, task_description, additional_info):
            if not task_description.strip():
                return gr.Textbox(value="Task description cannot be empty", visible=True), gr.Dropdown(choices=[])
            
            # Use provided task name or generate from description if empty
            final_task_name = task_name.strip() if task_name.strip() else task_description.split("\n")[0][:50]
            if len(final_task_name) >= 50:
                final_task_name = final_task_name[:47] + "..."
            
            save_task(final_task_name, task_description, additional_info)
            
            # Update dropdown
            tasks = load_tasks()
            return gr.Textbox(value=f"Task '{final_task_name}' saved successfully"), gr.Dropdown(choices=[task["name"] for task in tasks])
        
        # Connect event handlers for task templates
        refresh_templates_button.click(
            fn=update_task_templates_dropdown,
                    inputs=[],
            outputs=[task_template_dropdown]
        )
        
        task_template_dropdown.change(
            fn=load_task_template,
            inputs=[task_template_dropdown],
            outputs=[task_name_input, task, add_infos]
        )
        
        save_task_button.click(
            fn=save_current_task,
            inputs=[task_name_input, task, add_infos],
            outputs=[task_feedback, task_template_dropdown]
        )
        
        # Initialize task templates dropdown on page load
        demo.load(
            fn=update_task_templates_dropdown,
            inputs=[],
            outputs=[task_template_dropdown]
        )

        # Function to update the process log data with filtering
        def update_process_log_data(category="all", level="ALL", search_term=""):
            """
            Update the process log display with filtered data.
            
            Args:
                category: Category to filter logs by
                level: Log level to filter by
                search_term: Text to search for within logs
                
            Returns:
                Tuple of (filtered_logs, log_stats)
            """
            try:
                # Helper function for sanitizing inputs locally
                def safe_sanitize(value, default=None):
                    """Local sanitize function to avoid reference errors"""
                    # Handle None or empty values
                    if value is None:
                        return default
                        
                    # Handle list values from Gradio dropdowns
                    if isinstance(value, list):
                        if len(value) > 0:
                            # Get the first item for single-select dropdowns
                            value = value[0]
                        else:
                            return default
                    
                    # Handle empty strings
                    if isinstance(value, str) and not value.strip():
                        return default
                        
                    # Return the value with proper type conversion for common types
                    if isinstance(value, (int, float)):
                        return value
                    elif isinstance(value, str):
                        return value
                    else:
                        # Safe conversion to string for other types
                        try:
                            return str(value)
                        except:
                            return default
                
                # Sanitize inputs
                category_str = safe_sanitize(category, default="all")
                level_str = safe_sanitize(level, default="ALL")
                search_term_str = safe_sanitize(search_term, default="")
                
                # Get filtered logs
                filtered_logs = ""
                if search_term_str:
                    # Search term takes precedence
                    filtered_logs = data_log_handler.search_logs(search_term_str)
                elif level_str != "ALL":
                    # Then filter by level
                    filtered_logs = data_log_handler.get_logs_by_level(level_str)
                else:
                    # Otherwise filter by category
                    filtered_logs = data_log_handler.get_logs_by_category(category_str)
                
                # Get updated stats
                log_stats = data_log_handler.get_stats()
                
                return filtered_logs, log_stats
                
            except Exception as e:
                logging.error(f"Error updating process log data: {str(e)}")
                return f"Error updating logs: {str(e)}", json.dumps({"error": str(e)})
        
        # Create a tab for viewing agent activity and logs
        with gr.Tab("Agent Activity"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Log filtering controls
                    log_category_dropdown = gr.Dropdown(
                        choices=['all', 'agent_action', 'agent_thought', 'agent_memory', 'agent_planning', 
                                'agent_evaluation', 'agent_task_progress', 'extracted_data', 'error', 'system', 'other'],
                        value='all',
                        label="Log Category",
                        info="Filter logs by category"
                    )
                    
                    log_level_dropdown = gr.Dropdown(
                        choices=['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        value='ALL',
                        label="Log Level",
                        info="Filter logs by level"
                    )
                    
                    log_search_input = gr.Textbox(
                        label="Search Logs",
                        placeholder="Enter search term",
                        info="Search logs for specific text"
                    )
                    
                    update_logs_button = gr.Button("Refresh Logs", variant="primary")
                    clear_logs_button = gr.Button("Clear Logs", variant="secondary")
                    
                    log_stats_json = gr.JSON(
                        label="Log Statistics",
                        value={},
                        visible=True
                    )
                    
                with gr.Column(scale=3):
                    agent_log_display = gr.Textbox(
                        label="Agent Activity Log",
                        placeholder="Agent logs will appear here",
                        lines=30,
                        max_lines=50,
                        interactive=False
                    )
                    
                    export_logs_button = gr.Button("Export Logs")
            
            # Function to update agent logs
            def update_agent_logs(category, level, search_term):
                """
                Update the agent log display with filtered data.
                
                Args:
                    category: Category to filter logs by
                    level: Log level to filter by
                    search_term: Text to search for within logs
                    
                Returns:
                    Tuple of (filtered_logs, log_stats)
                """
                try:
                    # Helper function for sanitizing inputs locally
                    def safe_sanitize(value, default=None):
                        """Local sanitize function to avoid reference errors"""
                        # Handle None or empty values
                        if value is None:
                            return default
                            
                        # Handle list values from Gradio dropdowns
                        if isinstance(value, list):
                            if len(value) > 0:
                                # Get the first item for single-select dropdowns
                                value = value[0]
                            else:
                                return default
                        
                        # Handle empty strings
                        if isinstance(value, str) and not value.strip():
                            return default
                            
                        # Return the value with proper type conversion for common types
                        if isinstance(value, (int, float)):
                            return value
                        elif isinstance(value, str):
                            return value
                        else:
                            # Safe conversion to string for other types
                            try:
                                return str(value)
                            except:
                                return default
                    
                    # Sanitize inputs 
                    category_str = safe_sanitize(category, default="all")
                    level_str = safe_sanitize(level, default="ALL")
                    search_term_str = safe_sanitize(search_term, default="")
                    
                    # Get filtered logs
                    filtered_logs = ""
                    if search_term_str:
                        # Search term takes precedence
                        filtered_logs = data_log_handler.search_logs(search_term_str)
                    elif level_str != "ALL":
                        # Then filter by level
                        filtered_logs = data_log_handler.get_logs_by_level(level_str)
                    else:
                        # Otherwise filter by category
                        filtered_logs = data_log_handler.get_logs_by_category(category_str)
                    
                    # Get updated stats
                    log_stats = data_log_handler.get_stats()
                    
                    return filtered_logs, log_stats
                except Exception as e:
                    logging.error(f"Error updating agent logs: {str(e)}")
                    return f"Error updating logs: {str(e)}", json.dumps({"error": str(e)})
            
            # Function to clear agent logs
            def clear_agent_logs():
                """Clear all agent logs."""
                data_log_handler.clear_logs()
                return "", json.dumps({})
            
            # Function to export agent logs
            def export_agent_logs(category, level, search_term):
                """
                Export agent logs to a file.
                
                Args:
                    category: Category to filter logs by
                    level: Log level to filter by
                    search_term: Text to search for within logs
                    
                Returns:
                    String with export status message
                """
                try:
                    # Get filtered logs
                    logs, _ = update_agent_logs(category, level, search_term)
                    
                    # Create a user downloads directory if it doesn't exist
                    user_downloads = os.path.expanduser("~/Downloads/browser-use-exports")
                    os.makedirs(user_downloads, exist_ok=True)
                    
                    # Generate filename with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"agent_logs_{timestamp}.txt"
                    filepath = os.path.join(user_downloads, filename)
                    
                    # Write logs to file
                    with open(filepath, 'w') as f:
                        f.write(logs)
                    
                    return f"Logs exported to: {filepath}"
                except Exception as e:
                    logging.error(f"Error exporting agent logs: {str(e)}")
                    return f"Error exporting logs: {str(e)}"
                
            def export_comprehensive_logs():
                """
                Export comprehensive logs including terminal output, agent actions, web page info,
                and all other relevant data into a single file.
                
                This function combines data from:
                1. Agent tracker (terminal logs, actions, thoughts, errors)
                2. Process logs (all system and agent logs)
                3. Extracted data (web page content, URLs, etc.)
                
                Returns:
                    String with export status message
                """
                try:
                    # Create a user downloads directory if it doesn't exist
                    user_downloads = os.path.expanduser("~/Downloads/browser-use-exports")
                    os.makedirs(user_downloads, exist_ok=True)
                    
                    # Generate filename with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"comprehensive_logs_{timestamp}.txt"
                    filepath = os.path.join(user_downloads, filename)
                    
                    # Get agent tracker data
                    tracker = get_agent_tracker()
                    
                    # Create sections for the comprehensive log
                    sections = []
                    
                    # Add header
                    sections.append("=" * 80)
                    sections.append("COMPREHENSIVE BROWSER-USE AGENT LOG EXPORT")
                    sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    sections.append("=" * 80)
                    sections.append("")
                    
                    # Add process logs (all logs from the system)
                    sections.append("=" * 80)
                    sections.append("PROCESS LOGS (CHRONOLOGICAL)")
                    sections.append("=" * 80)
                    process_logs = data_log_handler.get_process_logs()
                    sections.append(process_logs)
                    sections.append("")
                    
                    # Add agent tracker data if available
                    if tracker:
                        # Add agent metrics
                        sections.append("=" * 80)
                        sections.append("AGENT METRICS")
                        sections.append("=" * 80)
                        metrics = tracker.get_current_metrics()
                        sections.append(json.dumps(metrics, indent=2))
                        sections.append("")
                        
                        # Add performance summary
                        sections.append("=" * 80)
                        sections.append("AGENT PERFORMANCE SUMMARY")
                        sections.append("=" * 80)
                        performance = tracker.get_performance_summary()
                        sections.append(json.dumps(performance, indent=2))
                        sections.append("")
                        
                        # Add sequence analysis
                        sections.append("=" * 80)
                        sections.append("AGENT SEQUENCE ANALYSIS")
                        sections.append("=" * 80)
                        sequence = tracker.get_sequence_analysis()
                        sections.append(json.dumps(sequence, indent=2))
                        sections.append("")
                        
                        # Add terminal logs
                        sections.append("=" * 80)
                        sections.append("TERMINAL LOGS")
                        sections.append("=" * 80)
                        sections.append("\n".join(tracker.terminal_logs))
                        sections.append("")
                        
                        # Add action history
                        sections.append("=" * 80)
                        sections.append("ACTION HISTORY")
                        sections.append("=" * 80)
                        sections.append(json.dumps(tracker.action_history, indent=2))
                        sections.append("")
                        
                        # Add thought history
                        sections.append("=" * 80)
                        sections.append("THOUGHT HISTORY")
                        sections.append("=" * 80)
                        sections.append(json.dumps(tracker.thought_history, indent=2))
                        sections.append("")
                        
                        # Add error history
                        sections.append("=" * 80)
                        sections.append("ERROR HISTORY")
                        sections.append("=" * 80)
                        sections.append(json.dumps(tracker.error_history, indent=2))
                        sections.append("")
                    
                    # Add extracted web data
                    sections.append("=" * 80)
                    sections.append("EXTRACTED WEB DATA")
                    sections.append("=" * 80)
                    web_data = extracted_data_collector.get_data_by_type("webpage")
                    if web_data:
                        # Format web data for better readability
                        for item in web_data:
                            sections.append(f"URL: {item.get('url', 'N/A')}")
                            sections.append(f"Title: {item.get('title', 'N/A')}")
                            sections.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.get('timestamp', 0)))}")
                            sections.append(f"Source: {item.get('source', 'N/A')}")
                            
                            # Add HTML content if available (truncated for readability)
                            if 'html' in item:
                                html_content = item['html']
                                if len(html_content) > 1000:
                                    html_content = html_content[:1000] + "... [truncated]"
                                sections.append("HTML Content:")
                                sections.append(html_content)
                            
                            # Add text content if available
                            if 'text' in item:
                                sections.append("Text Content:")
                                sections.append(item['text'])
                            
                            sections.append("-" * 40)
                    else:
                        sections.append("No web data extracted")
                    sections.append("")
                    
                    # Add HTML content
                    sections.append("=" * 80)
                    sections.append("HTML CONTENT")
                    sections.append("=" * 80)
                    html_data = extracted_data_collector.get_data_by_type("html")
                    if html_data:
                        for i, item in enumerate(html_data):
                            sections.append(f"HTML Item #{i+1}")
                            sections.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.get('timestamp', 0)))}")
                            sections.append(f"Source: {item.get('source', 'N/A')}")
                            
                            # Add HTML content (truncated for readability)
                            if 'html' in item:
                                html_content = item['html']
                                if len(html_content) > 1000:
                                    html_content = html_content[:1000] + "... [truncated]"
                                sections.append("HTML Content:")
                                sections.append(html_content)
                            
                            sections.append("-" * 40)
                    else:
                        sections.append("No HTML content extracted")
                    sections.append("")
                    
                    # Add all extracted data metadata
                    sections.append("=" * 80)
                    sections.append("EXTRACTED DATA METADATA")
                    sections.append("=" * 80)
                    metadata = extracted_data_collector.get_metadata()
                    sections.append(json.dumps(metadata, indent=2))
                    sections.append("")
                    
                    # Add a JSON file with complete data for programmatic access
                    json_filename = f"comprehensive_logs_{timestamp}.json"
                    json_filepath = os.path.join(user_downloads, json_filename)
                    
                    # Prepare complete data object
                    complete_data = {
                        "process_logs": data_log_handler.process_logs,
                        "agent_tracker": tracker.export_to_json() if tracker else None,
                        "extracted_data": extracted_data_collector.get_data(),
                        "metadata": {
                            "export_time": time.time(),
                            "export_time_formatted": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "process_log_count": len(data_log_handler.process_logs),
                            "extracted_data_count": len(extracted_data_collector.get_data())
                        }
                    }
                    
                    # Write JSON data
                    with open(json_filepath, 'w') as f:
                        json.dump(complete_data, indent=2, fp=f)
                    
                    # Write the comprehensive log to file
                    with open(filepath, 'w') as f:
                        f.write("\n".join(sections))
                    
                    return f"Comprehensive logs exported to: {filepath}\nJSON data exported to: {json_filepath}"
                except Exception as e:
                    logging.error(f"Error exporting comprehensive logs: {str(e)}")
                    return f"Error exporting comprehensive logs: {str(e)}"
        # Function to update extracted data based on filters
        def update_extracted_data(data_type, data_source, search_term):
            """Return filtered extracted data along with updated dropdown choices for types and sources."""
            try:
                # Get all data and metadata
                all_data = extracted_data_collector.get_data()
                metadata = extracted_data_collector.get_metadata()
                
                # If no filters are applied, use all data
                if data_type == "all" and data_source == "all" and not search_term.strip():
                    filtered_data = all_data
                else:
                    filtered_data = all_data
                    if data_type != "all":
                        filtered_data = [item for item in filtered_data if item.get("type") == data_type]
                    if data_source != "all":
                        filtered_data = [item for item in filtered_data if item.get("source") == data_source]
                    if search_term.strip():
                        st = search_term.lower()
                        filtered_data = [item for item in filtered_data if st in str(item).lower()]
                
                # Prepare dropdown updates based on metadata
                type_update = gr.Dropdown.update(choices=metadata.get("types", []))
                source_update = gr.Dropdown.update(choices=metadata.get("sources", []))
                
                return filtered_data, metadata, type_update, source_update
            except Exception as e:
                logging.error(f"Error in update_extracted_data: {str(e)}")
                return [], {"types": [], "sources": []}, gr.Dropdown.update(choices=[]), gr.Dropdown.update(choices=[])
        
        # Connect the update button to the extracted data with filtering
        update_extracted_data_button.click(
            fn=update_extracted_data,
            inputs=[data_type_dropdown, data_source_dropdown, data_search_input],
            outputs=[extracted_data_log, data_stats_json, data_type_dropdown, data_source_dropdown]
        )
        
        # Connect the dropdowns and search to auto-update
        data_type_dropdown.change(
            fn=update_extracted_data,
            inputs=[data_type_dropdown, data_source_dropdown, data_search_input],
            outputs=[extracted_data_log, data_stats_json, data_type_dropdown, data_source_dropdown]
        )
        
        data_source_dropdown.change(
            fn=update_extracted_data,
            inputs=[data_type_dropdown, data_source_dropdown, data_search_input],
            outputs=[extracted_data_log, data_stats_json, data_type_dropdown, data_source_dropdown]
        )
        
        data_search_input.submit(
            fn=update_extracted_data,
            inputs=[data_type_dropdown, data_source_dropdown, data_search_input],
            outputs=[extracted_data_log, data_stats_json, data_type_dropdown, data_source_dropdown]
        )
        
        # Function to export extracted data
        def export_extracted_data(data_type, data_source, search_term):
            # Get filtered data
            filtered_data, _ = update_extracted_data(data_type, data_source, search_term)
            
            try:
                # Create a timestamp for the filename
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create a user downloads directory if it doesn't exist
                user_downloads = os.path.expanduser("~/Downloads/browser-use-exports")
                os.makedirs(user_downloads, exist_ok=True)
                
                # Create a more descriptive filename
                filename_parts = []
                if data_type and data_type != "all":
                    filename_parts.append(data_type)
                if data_source and data_source != "all":
                    filename_parts.append(data_source)
                if search_term:
                    # Limit search term length in filename
                    safe_term = search_term[:20].replace(" ", "_")
                    if safe_term:
                        filename_parts.append(safe_term)
                
                # Default to "full_export" if no specific filters
                base_name = "_".join(filename_parts) if filename_parts else "full_export"
                export_path = os.path.join(user_downloads, f"{base_name}_{timestamp}.json")
                
                # Ensure we're not truncating any webpage content
                for item in filtered_data:
                    # Make sure we're preserving full HTML content
                    if "html_content" in item and isinstance(item["html_content"], str) and len(item["html_content"]) > 1000000:
                        # Log that we're preserving large HTML content
                        logging.info(f"Preserving large HTML content ({len(item['html_content'])} bytes) in export")
                
                # Write the data to the file with pretty formatting for readability
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                
                # Log the export for tracking
                logging.info(f"Exported {len(filtered_data)} items to {export_path}")
                
                return f"Extracted data exported to: {export_path}"
            except Exception as e:
                logging.error(f"Error exporting data: {str(e)}")
                return f"Error exporting extracted data: {str(e)}"
        
        # Function to clear extracted data
        def clear_extracted_data():
            extracted_data_collector.clear_data()
            return [], extracted_data_collector.get_metadata()
        
        # Bind the export and clear buttons
        export_data_button.click(
            fn=export_extracted_data,
            inputs=[data_type_dropdown, data_source_dropdown, data_search_input],
            outputs=[gr.Textbox(label="Export Status")]
        )
        
        clear_data_button.click(
            fn=clear_extracted_data,
            inputs=[],
            outputs=[extracted_data_log, data_stats_json, data_type_dropdown, data_source_dropdown]
        )
        
        # Update the extracted data when the page loads
        demo.load(
            fn=update_extracted_data,
            inputs=[data_type_dropdown, data_source_dropdown, data_search_input],
            outputs=[extracted_data_log, data_stats_json, data_type_dropdown, data_source_dropdown]
        )
        
        # Add JavaScript to update the extracted data periodically
        js_update_data = """
        <script>
        // Function to update the extracted data
        function updateExtractedData() {
            // Find the update button by its text content
            var buttons = document.querySelectorAll('button');
            var updateButton = null;
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.includes('Refresh Data')) {
                    updateButton = buttons[i];
                    break;
                }
            }
            
            if (updateButton) {
                updateButton.click();
            } else {
                console.log('Update data button not found');
            }
            
            // Schedule the next update
            setTimeout(updateExtractedData, 5000);
        }
        
        // Start updating the extracted data when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Wait a bit for the UI to initialize
            setTimeout(updateExtractedData, 5000);
        });
        </script>
        """
        
        # Add the JavaScript to the page
        gr.HTML(js_update_data)

        # Add function to export agent history
        def export_agent_history(history_file_path):
            if not history_file_path or not os.path.exists(history_file_path):
                return "No agent history file available to export."
            
            try:
                # Get the filename from the path
                filename = os.path.basename(history_file_path)
                
                # Create a user downloads directory if it doesn't exist
                user_downloads = os.path.expanduser("~/Downloads/browser-use-exports")
                os.makedirs(user_downloads, exist_ok=True)
                
                # Copy the file to the user's downloads directory
                export_path = os.path.join(user_downloads, filename)
                shutil.copy2(history_file_path, export_path)
                
                return f"Agent history exported to: {export_path}"
            except Exception as e:
                return f"Error exporting agent history: {str(e)}"
        
        # Bind the export button click event
        export_history_button.click(
            fn=export_agent_history,
            inputs=[agent_history_file],
            outputs=[gr.Textbox(label="Export Status")]
        )

        # Run button click handler
        run_button.click(
            fn=run_with_stream,
                inputs=[
                    agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                    use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                    save_recording_path, save_agent_history_path, save_trace_path,  # Include the new path
                    enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step, tool_calling_method
                ],
            outputs=[
        stream_output,           # Browser view
                final_result_output,    # Final result
                errors_output,          # Errors
                model_actions_output,   # Model actions
                model_thoughts_output,  # Model thoughts
                recording_display,      # Latest recording
                trace_file,             # Trace file
                agent_history_file,     # Agent history file
                stop_button,            # Stop button
        run_button,             # Run button
        process_log_data         # Process log (added)
    ],
)
        
        # Also update the process log when the run button is clicked
        run_button.click(
            fn=lambda: data_log_handler.get_process_logs(),
            inputs=[],
            outputs=[process_log_data]
        )

        # Bind the stop button click event after errors_output is defined
        stop_button.click(
            fn=stop_agent,
                    inputs=[],
            outputs=[errors_output, stop_button, run_button],
        )

        # Also update the process log when the stop button is clicked
        stop_button.click(
            fn=lambda: data_log_handler.get_process_logs(),
            inputs=[],
            outputs=[process_log_data]
                )

        with gr.TabItem("🎥 Recordings", id=8):
            def list_recordings(save_recording_path):
                if not os.path.exists(save_recording_path):
                    return []

                # Get all video files
                recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                # Sort recordings by creation time (oldest first)
                recordings.sort(key=os.path.getctime)

                # Add numbering to the recordings
                numbered_recordings = []
                for idx, recording in enumerate(recordings, start=1):
                    filename = os.path.basename(recording)
                    numbered_recordings.append((recording, f"{idx}. {filename}"))

                return numbered_recordings

            recordings_gallery = gr.Gallery(
                label="Recordings",
                value=list_recordings(config['save_recording_path']),
                columns=3,
                height="auto",
                object_fit="contain"
            )

            refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
            refresh_button.click(
                fn=list_recordings,
                inputs=save_recording_path,
                outputs=recordings_gallery
            )

    return demo

def parse_args():
    """
    Parse command-line arguments for the web UI.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Browser Use Web UI")
    parser.add_argument("--theme", type=str, default="Ocean", help="Theme for the UI")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    
    return parser.parse_args()

def find_available_port(start_port=7790, max_attempts=10):
    """
    Find an available port starting from the given port.
    
    Args:
        start_port: The port to start checking from
        max_attempts: Maximum number of ports to check
        
    Returns:
        int: An available port or None if none found
    """
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:  # Port is available
                return port
        except:
            sock.close()
    
    return None  # No available ports found

def main():
    """Main function to start the web UI server"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config_dict = default_config()
    if os.path.exists(args.config):
        loaded_config = load_config_from_file(args.config)
        if isinstance(loaded_config, dict):
            config_dict = loaded_config
    
    # Ensure add_infos is in the config
    if 'add_infos' not in config_dict:
        config_dict['add_infos'] = ""
    
    # Fix any broken playlist files
    playlist_dir = "saved_playlists"
    os.makedirs(playlist_dir, exist_ok=True)
    
    for playlist_file in glob.glob(os.path.join(playlist_dir, "*.json")):
        try:
            # Try to load the file as JSON
            with open(playlist_file, 'r') as f:
                json.loads(f.read())
        except json.JSONDecodeError:
            # File is broken, attempt to fix it
            logging.warning(f"Found corrupted playlist file: {playlist_file}, attempting to fix...")
            if fix_broken_playlist_file(playlist_file):
                logging.info(f"Successfully fixed playlist file: {playlist_file}")
            else:
                logging.error(f"Failed to fix playlist file: {playlist_file}")

    # Create the UI
    demo = create_ui(config_dict, theme_name=args.theme)
    
    # Find an available port
    port = args.port
    if not is_port_available(port):
        # Try to find any available port in a wider range
        available_port = find_available_port(start_port=7790, max_attempts=20)
        if available_port:
            logging.warning(f"Port {port} is busy, using port {available_port} instead")
            port = available_port
        else:
            # Last resort - try port 0 which lets the OS assign a free port
            logging.warning("No specific available ports found, letting OS choose a port")
            port = 0
    
    # Launch the server with improved error handling
    try:
        demo.launch(server_name=args.ip, server_port=port)
    except OSError as e:
        if "Cannot find empty port" in str(e):
            # Final attempt with completely different range
            fallback_port = find_available_port(start_port=8800, max_attempts=50)
            if fallback_port:
                logging.warning(f"Using fallback port {fallback_port}")
                demo.launch(server_name=args.ip, server_port=fallback_port)
            else:
                logging.error("Could not find any available ports")
                sys.exit(1)
        else:
            logging.error(f"Error launching server: {str(e)}")
            raise e

def is_port_available(port):
    """Check if a port is available."""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result != 0  # Port is available if connection fails
    except:
        sock.close()
        return False

if __name__ == '__main__':
    main()

def sanitize_gradio_input(value, default=None):
    """
    Utility function to sanitize inputs from Gradio components.
    Handles conversion from lists to single values and ensures proper string conversion.
    
    Args:
        value: The input value from a Gradio component
        default: Default value if the input is None or empty
        
    Returns:
        Sanitized value suitable for use in backend functions
    """
    try:
        # Handle None or empty values
        if value is None:
            return default
            
        # Handle list values from Gradio dropdowns
        if isinstance(value, list):
            if len(value) > 0:
                # Get the first item for single-select dropdowns
                value = value[0]
            else:
                return default
        
        # Handle empty strings
        if isinstance(value, str) and not value.strip():
            return default
            
        # Return the value with proper type conversion for common types
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, str):
            return value
        else:
            # Safe conversion to string for other types
            try:
                return str(value)
            except:
                return default
    except Exception as e:
        logging.error(f"Error sanitizing input: {str(e)}")
        return default

def fix_broken_playlist_file(file_path):
    """
    Attempts to fix a broken playlist file by reading as much as possible
    and creating a valid JSON structure if the file is corrupted.
    
    Args:
        file_path: Path to the playlist file
        
    Returns:
        bool: True if fixed successfully, False otherwise
    """
    try:
        # Try to open and read the file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Try to parse it as JSON
        try:
            json.loads(content)
            # If we get here, the file is valid JSON
            return False  # No fixing needed
        except json.JSONDecodeError:
            # File is broken, create a minimal valid structure
            playlist_name = os.path.basename(file_path).replace('.json', '')
            fixed_content = {
                "name": playlist_name,
                "description": f"Auto-fixed playlist (previously corrupted)",
                "tasks": []
            }
            
            # Write the fixed content back to the file
            with open(file_path, 'w') as f:
                json.dump(fixed_content, f, indent=2)
                
            logging.info(f"Fixed broken playlist file: {file_path}")
            return True
            
    except Exception as e:
        logging.error(f"Failed to fix playlist file {file_path}: {str(e)}")
        return False

# Agent tracker for advanced monitoring and analytics
class AgentTracker:
    """
    Advanced tracker for agent analytics and metrics
    
    Captures detailed information about agent performance including:
    - Success/failure rates of actions
    - Time spent on different types of operations
    - Error patterns and frequencies
    - Memory evolution throughout agent execution
    - Task completion metrics
    """
    
    def __init__(self):
        """Initialize the agent tracker with empty metrics and tracking state"""
        self.metrics = {
            "actions": {"total": 0, "successful": 0, "failed": 0, "unknown": 0},
            "errors": {},
            "thoughts": 0,
            "memory_updates": 0,
            "evaluations": 0,
            "plans": 0,
            "time_spent": {"total": 0, "thinking": 0, "acting": 0, "waiting": 0}
        }
        self.active = False
        self.start_time = None
        self.terminal_logs = []  # Store terminal logs for inclusion in the export
        self.end_time = None
        self.step_history = []
        self.action_history = []
        self.thought_history = []
        self.memory_history = []
        self.evaluation_history = []
        self.plan_history = []
        self.error_history = []
        self.lock = threading.RLock()
    
    def add_terminal_log(self, log_entry):
        """Add a terminal log entry to the tracker"""
        if isinstance(log_entry, str):
            self.terminal_logs.append(log_entry)
        
    def export_to_json(self, filepath=None):
        """
        Export tracking data to JSON
        
        Args:
            filepath (str, optional): Path to save JSON file
            
        Returns:
            str: JSON string if no filepath provided
        """
        data = {
            "metrics": self.get_current_metrics(),
            "performance_summary": self.get_performance_summary(),
            "sequence_analysis": self.get_sequence_analysis(),
            "step_history": self.step_history,
            "action_history": self.action_history,
            "thought_history": self.thought_history,
            "error_history": self.error_history,
            "terminal_logs": self.terminal_logs,  # Include terminal logs in export
            "start_time": self.start_time,
            "end_time": self.end_time
        }
        
        json_data = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_data)
            return filepath
            
        return json_data
    
    def start_tracking(self):
        """Start tracking agent performance"""
        self.active = True
        self.start_time = time.time()
        
    def stop_tracking(self):
        """Stop tracking agent performance"""
        self.active = False
        self.end_time = time.time()
        
    def reset(self):
        """Reset all tracking data"""
        self.__init__()
        
    def track_action(self, action_data):
        """
        Track an agent action
        
        Args:
            action_data (dict): Contains action information including:
                - type: Action type (web_interaction, API_call, etc.)
                - details: Specific action details
                - status: Success/failure status
                - result: Action result
                - timestamp: When action occurred
        """
        if not self.active:
            return
            
        with self.lock:
            current_step = self._get_current_step()
            
            # Normalize the action data
            if isinstance(action_data, str):
                action_data = {"type": "unknown", "details": action_data, "status": "unknown"}
                
            if not isinstance(action_data, dict):
                action_data = {"type": "unknown", "details": str(action_data), "status": "unknown"}
                
            # Ensure required fields
            if "timestamp" not in action_data:
                action_data["timestamp"] = str(time.time())
                
            if "status" not in action_data:
                action_data["status"] = "unknown"
                
            # Add step information
            action_data["step"] = str(current_step)
                
            # Update metrics
            self.metrics["actions"]["total"] += 1
            status = action_data.get("status", "unknown").lower()
            
            if status == "success":
                self.metrics["actions"]["successful"] += 1
            elif status == "failed" or status == "failure" or status == "error":
                self.metrics["actions"]["failed"] += 1
            else:
                self.metrics["actions"]["unknown"] += 1
                
            # Store in history
            self.action_history.append(action_data)
            
            # Update step history
            if current_step > 0 and len(self.step_history) >= current_step:
                if "actions" not in self.step_history[current_step-1]:
                    self.step_history[current_step-1]["actions"] = []
                self.step_history[current_step-1]["actions"].append(action_data)
            
            return True
            
    def track_thought(self, thought_data):
        """
        Track an agent thought/reasoning process
        
        Args:
            thought_data (dict): Contains thought information including:
                - content: The thought content
                - category: Type of thinking (planning, evaluation, etc.)
                - timestamp: When thought occurred
        """
        if not self.active:
            return
            
        with self.lock:
            current_step = self._get_current_step()
            
            # Normalize the thought data
            if isinstance(thought_data, str):
                thought_data = {"content": thought_data, "category": "general"}
                
            if not isinstance(thought_data, dict):
                thought_data = {"content": str(thought_data), "category": "general"}
                
            # Ensure required fields
            if "timestamp" not in thought_data:
                thought_data["timestamp"] = str(time.time())
                
            if "category" not in thought_data:
                thought_data["category"] = "general"
                
            # Add step information
            thought_data["step"] = str(current_step)
                
            # Update metrics
            self.metrics["thoughts"] += 1
                
            # Store in history
            self.thought_history.append(thought_data)
            
            # Update step history
            if current_step > 0 and len(self.step_history) >= current_step:
                if "thoughts" not in self.step_history[current_step-1]:
                    self.step_history[current_step-1]["thoughts"] = []
                self.step_history[current_step-1]["thoughts"].append(thought_data)
            
            return True
            
    def track_memory(self, memory_data):
        """
        Track a memory operation or update
        
        Args:
            memory_data (dict): Contains memory information including:
                - operation: Type of memory operation (add, update, retrieve)
                - content: The memory content
                - timestamp: When memory operation occurred
        """
        if not self.active:
            return
            
        with self.lock:
            current_step = self._get_current_step()
            
            # Normalize the memory data
            if isinstance(memory_data, str):
                memory_data = {"content": memory_data, "operation": "update"}
                
            if not isinstance(memory_data, dict):
                memory_data = {"content": str(memory_data), "operation": "update"}
                
            # Ensure required fields
            if "timestamp" not in memory_data:
                memory_data["timestamp"] = str(time.time())
                
            if "operation" not in memory_data:
                memory_data["operation"] = "update"
                
            # Add step information
            memory_data["step"] = str(current_step)
                
            # Update metrics
            self.metrics["memory_updates"] += 1
                
            # Store in history
            self.memory_history.append(memory_data)
            
            # Update step history
            if current_step > 0 and len(self.step_history) >= current_step:
                if "memory" not in self.step_history[current_step-1]:
                    self.step_history[current_step-1]["memory"] = []
                self.step_history[current_step-1]["memory"].append(memory_data)
            
            return True
            
    def track_evaluation(self, eval_data):
        """
        Track an agent's self-evaluation
        
        Args:
            eval_data (dict): Contains evaluation information including:
                - aspect: What's being evaluated
                - rating: Numerical or categorical rating
                - reasoning: Reasoning behind evaluation
                - timestamp: When evaluation occurred
        """
        if not self.active:
            return
            
        with self.lock:
            current_step = self._get_current_step()
            
            # Normalize the evaluation data
            if isinstance(eval_data, str):
                eval_data = {"reasoning": eval_data, "aspect": "general", "rating": "unknown"}
                
            if not isinstance(eval_data, dict):
                eval_data = {"reasoning": str(eval_data), "aspect": "general", "rating": "unknown"}
                
            # Ensure required fields
            if "timestamp" not in eval_data:
                eval_data["timestamp"] = str(time.time())
                
            if "aspect" not in eval_data:
                eval_data["aspect"] = "general"
                
            if "rating" not in eval_data:
                eval_data["rating"] = "unknown"
                
            # Add step information
            eval_data["step"] = str(current_step)
                
            # Update metrics
            self.metrics["evaluations"] += 1
                
            # Store in history
            self.evaluation_history.append(eval_data)
            
            # Update step history
            if current_step > 0 and len(self.step_history) >= current_step:
                if "evaluations" not in self.step_history[current_step-1]:
                    self.step_history[current_step-1]["evaluations"] = []
                self.step_history[current_step-1]["evaluations"].append(eval_data)
            
            return True
            
    def track_plan(self, plan_data):
        """
        Track an agent's planning process
        
        Args:
            plan_data (dict): Contains plan information including:
                - steps: List of planned steps
                - goal: Plan objective
                - timestamp: When plan was created
        """
        if not self.active:
            return
            
        with self.lock:
            current_step = self._get_current_step()
            
            # Normalize the plan data
            if isinstance(plan_data, str):
                plan_data = {"steps": [plan_data], "goal": "unknown"}
                
            if not isinstance(plan_data, dict):
                plan_data = {"steps": [str(plan_data)], "goal": "unknown"}
                
            # Ensure required fields
            if "timestamp" not in plan_data:
                plan_data["timestamp"] = str(time.time())
                
            # Add step information
            plan_data["step"] = str(current_step)
                
            # Update metrics
            self.metrics["plans"] += 1
                
            # Store in history
            self.plan_history.append(plan_data)
            
            # Update step history
            if current_step > 0 and len(self.step_history) >= current_step:
                if "plans" not in self.step_history[current_step-1]:
                    self.step_history[current_step-1]["plans"] = []
                self.step_history[current_step-1]["plans"].append(plan_data)
            
            return True
            
    def track_error(self, error_data):
        """
        Track an error encountered by the agent
        
        Args:
            error_data (dict): Contains error information including:
                - type: Error type/category
                - message: Error message
                - context: What caused the error
                - timestamp: When error occurred
        """
        if not self.active:
            return
            
        with self.lock:
            current_step = self._get_current_step()
            
            # Normalize the error data
            if isinstance(error_data, str):
                error_data = {"message": error_data, "type": "unknown", "context": "unknown"}
                
            if not isinstance(error_data, dict):
                error_data = {"message": str(error_data), "type": "unknown", "context": "unknown"}
                
            # Ensure required fields
            if "timestamp" not in error_data:
                error_data["timestamp"] = str(time.time())
                
            if "type" not in error_data:
                error_data["type"] = "unknown"
                
            # Add step information
            error_data["step"] = str(current_step)
                
            # Update error metrics
            error_type = error_data.get("type", "unknown")
            if error_type in self.metrics["errors"]:
                self.metrics["errors"][error_type] += 1
            else:
                self.metrics["errors"][error_type] = 1
                
            # Store in history
            self.error_history.append(error_data)
            
            # Update step history
            if current_step > 0 and len(self.step_history) >= current_step:
                if "errors" not in self.step_history[current_step-1]:
                    self.step_history[current_step-1]["errors"] = []
                self.step_history[current_step-1]["errors"].append(error_data)
            
            return True
            
    def get_current_metrics(self):
        """Get current agent performance metrics"""
        self._update_metrics()
        return self.metrics
        
    def get_step_details(self, step):
        """
        Get detailed information about a specific step
        
        Args:
            step (int): Step number to retrieve
            
        Returns:
            dict: Step details or empty dict if step not found
        """
        if step <= 0 or step > len(self.step_history):
            return {}
            
        return self.step_history[step-1]
        
    def get_activity_timeline(self, start_time=None, end_time=None):
        """
        Get a timeline of agent activity within the specified timeframe
        
        Args:
            start_time (float, optional): Start timestamp
            end_time (float, optional): End timestamp
            
        Returns:
            list: Chronological list of agent activities
        """
        # Set default times if not provided
        if not start_time:
            start_time = self.start_time or 0
            
        if not end_time:
            end_time = self.end_time or time.time()
            
        # Combine all activities
        activities = []
        
        # Add actions
        for action in self.action_history:
            if start_time <= action.get("timestamp", 0) <= end_time:
                activities.append({
                    "type": "action",
                    "data": action,
                    "timestamp": action.get("timestamp", 0)
                })
                
        # Add thoughts
        for thought in self.thought_history:
            if start_time <= thought.get("timestamp", 0) <= end_time:
                activities.append({
                    "type": "thought",
                    "data": thought,
                    "timestamp": thought.get("timestamp", 0)
                })
                
        # Sort by timestamp
        activities.sort(key=lambda x: x.get("timestamp", 0))
        
        return activities
        
    def get_sequence_analysis(self):
        """
        Analyze agent behavior patterns and sequences
        
        Returns:
            dict: Analysis of agent behavior patterns
        """
        # Calculate basic pattern metrics
        action_thought_ratio = self.metrics["actions"]["total"] / max(1, self.metrics["thoughts"])
        
        # Count action-thought sequences
        action_after_thought = 0
        thought_after_action = 0
        
        timeline = self.get_activity_timeline()
        for i in range(1, len(timeline)):
            curr = timeline[i]["type"]
            prev = timeline[i-1]["type"]
            
            if curr == "action" and prev == "thought":
                action_after_thought += 1
                
            if curr == "thought" and prev == "action":
                thought_after_action += 1
                
        return {
            "action_thought_ratio": action_thought_ratio,
            "action_after_thought": action_after_thought,
            "thought_after_action": thought_after_action,
            "success_rate": self._calculate_success_rate(),
            "error_summary": self._get_error_summary()
        }
        
    def get_performance_summary(self):
        """
        Get a summary of agent performance metrics
        
        Returns:
            dict: Summary performance metrics
        """
        total_time = 0
        if self.start_time:
            end = self.end_time or time.time()
            total_time = end - self.start_time
            
        return {
            "total_time": total_time,
            "actions_per_minute": (self.metrics["actions"]["total"] * 60) / max(1, total_time),
            "success_rate": self._calculate_success_rate(),
            "error_rate": len(self.error_history) / max(1, self.metrics["actions"]["total"]),
            "total_steps": len(self.step_history),
            "total_actions": self.metrics["actions"]["total"],
            "total_thoughts": self.metrics["thoughts"],
            "total_errors": len(self.error_history)
        }
        
    def _update_metrics(self):
        """Update derived metrics based on current data"""
        # Calculate time spent
        if self.start_time:
            end = self.end_time or time.time()
            self.metrics["time_spent"]["total"] = end - self.start_time
            
        # Update success rate
        self.metrics["success_rate"] = self._calculate_success_rate()
        
        # Count steps
        self.metrics["steps"] = len(self.step_history)
            
    def _calculate_success_rate(self):
        """Calculate the success rate of actions"""
        total = self.metrics["actions"]["total"]
        if total == 0:
            return 0
            
        return self.metrics["actions"]["successful"] / total
        
    def _get_current_step(self):
        """Get the current step number"""
        return len(self.step_history)
        
    def _get_error_summary(self):
        """Get a summary of errors encountered"""
        # Count errors by type
        error_counts = {}
        for error in self.error_history:
            error_type = error.get("type", "unknown")
            if error_type in error_counts:
                error_counts[error_type] += 1
            else:
                error_counts[error_type] = 1
                
        # Get most common errors
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total": len(self.error_history),
            "by_type": error_counts,
            "most_common": sorted_errors[:3] if sorted_errors else []
        }


# Custom log interceptor to feed data to the agent tracker
class AgentTrackerLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        
    def emit(self, record):
        try:
            log_entry = self.format(record)
            
            # Only process agent-related logs
            if record.name != 'src.agent.custom_agent':
                return
                
            # Extract action data
            if "🛠️  Action" in log_entry:
                try:
                    # Parse action details from log
                    action_parts = log_entry.split("🛠️  Action")[1].strip()
                    action_num = action_parts.split("/")[0].strip()
                    action_content = action_parts.split(":", 1)[1].strip() if ":" in action_parts else action_parts
                    
                    # Track the action
                    get_agent_tracker().track_action({
                        "action_type": "browser_action",
                        "action_number": action_num,
                        "content": action_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking action: {str(e)}")
            
            # Extract thought data
            elif "🤔 Thought" in log_entry:
                try:
                    thought_content = log_entry.split("🤔 Thought:")[1].strip()
                    get_agent_tracker().track_thought({
                        "thought_type": "reasoning",
                        "content": thought_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking thought: {str(e)}")
            
            # Extract memory data
            elif "🧠 New Memory" in log_entry:
                try:
                    memory_content = log_entry.split("🧠 New Memory:")[1].strip()
                    get_agent_tracker().track_memory({
                        "memory_type": "new",
                        "content": memory_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking memory: {str(e)}")
            
            # Extract evaluation data
            elif "Eval:" in log_entry:
                try:
                    success = "✅" in log_entry
                    failed = "❌" in log_entry
                    eval_content = log_entry.split("Eval:")[1].strip()
                    
                    get_agent_tracker().track_evaluation({
                        "success": success,
                        "failed": failed,
                        "content": eval_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking evaluation: {str(e)}")
            
            # Extract planning data
            elif "📋 Future Plans" in log_entry:
                try:
                    plan_content = log_entry.split("📋 Future Plans:")[1].strip()
                    get_agent_tracker().track_plan({
                        "plan_type": "future",
                        "content": plan_content,
                        "raw_log": log_entry
                    })
                except Exception as e:
                    logging.error(f"Error tracking plan: {str(e)}")
            
            # Extract step information to track progress
            elif "📍 Step" in log_entry:
                try:
                    step_match = re.search(r"Step (\d+)", log_entry)
                    if step_match:
                        step_num = int(step_match.group(1))
                        get_agent_tracker().track_action({
                            "action_type": "step_transition",
                            "step": step_num,
                            "raw_log": log_entry
                        })
                except Exception as e:
                    logging.error(f"Error tracking step: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in AgentTrackerLogHandler: {str(e)}")
