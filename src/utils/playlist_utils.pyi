"""
Utility functions for managing task playlists.
"""

import os
import json
import glob
import logging
import datetime
import asyncio
import pandas as pd
import gradio as gr

logger = logging.getLogger(__name__)
from gradio.events import Dependency

# Custom DataFrameComponent to handle file path issues
class CustomDataFrameComponent(gr.Dataframe):
    """
    Custom DataFrameComponent that handles file path issues in postprocessing.
    """
    def postprocess(self, value):
        """
        Override the postprocess method to handle file path issues.
        
        Args:
            value: The value to postprocess
            
        Returns:
            pd.DataFrame: The postprocessed value
        """
        import pandas as pd
        
        # If value is a string (file path), check if it exists
        if isinstance(value, str):
            if not value or not os.path.exists(value):
                # Return an empty DataFrame with the correct columns
                return pd.DataFrame(columns=["Task Name", "Description"])
        
        # Otherwise, use the parent class's postprocess method
        try:
            return super().postprocess(value)
        except Exception as e:
            logger.error(f"Error in DataFrameComponent postprocessing: {str(e)}")
            # Return an empty DataFrame with the correct columns
            return pd.DataFrame(columns=["Task Name", "Description"])
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer

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

# Playlist UI helper functions
def load_saved_playlists_for_ui():
    """
    Load playlists for the UI dropdown.
    
    Returns:
        list: List of playlist names formatted for Gradio dropdown
    """
    try:
        playlists = load_playlists()
        # Format as list of strings for Gradio dropdown
        playlist_names = [playlist["name"] for playlist in playlists]
        logger.info(f"Loaded {len(playlist_names)} playlists for dropdown")
        return playlist_names
    except Exception as e:
        logger.error(f"Error loading playlists: {str(e)}")
        return []

def load_selected_playlist_for_ui(playlist_name):
    """
    Load a selected playlist in the UI.
    
    Args:
        playlist_name (str): Name of the playlist to load
        
    Returns:
        tuple: (task_list, playlist_name, description, feedback_message)
    """
    import pandas as pd
    
    if not playlist_name:
        logger.warning("No playlist name provided to load")
        # Return an empty DataFrame with the correct columns
        empty_df = pd.DataFrame(columns=["Task Name", "Description"])
        return empty_df, "", "", "No playlist selected"
    
    logger.info(f"Loading playlist: {playlist_name}")
    playlists = load_playlists()
    selected_playlist = next((p for p in playlists if p["name"] == playlist_name), None)
    
    if not selected_playlist:
        logger.warning(f"Playlist not found: {playlist_name}")
        # Return an empty DataFrame with the correct columns
        empty_df = pd.DataFrame(columns=["Task Name", "Description"])
        return empty_df, "", "", f"Playlist '{playlist_name}' not found"
    
    # Load the tasks in the playlist
    from src.utils.utils import load_tasks
    
    task_list = []
    tasks = load_tasks()
    
    # Log the tasks we're looking for and what's available
    logger.info(f"Playlist tasks: {selected_playlist['tasks']}")
    task_names = [t['name'] for t in tasks]
    logger.info(f"Available tasks: {task_names}")
    
    # If no tasks are found in the saved playlist, create a simple DataFrame
    if not selected_playlist.get("tasks", []):
        logger.warning(f"No tasks found in playlist '{playlist_name}'")
        # Return an empty DataFrame with the correct columns
        empty_df = pd.DataFrame(columns=["Task Name", "Description"])
        return empty_df, selected_playlist["name"], selected_playlist.get("description", ""), f"Loaded playlist: {playlist_name} (empty)"
    
    # Try to match tasks by name
    for task_name in selected_playlist.get("tasks", []):
        if not task_name:  # Skip empty task names
            continue
            
        # First try exact match
        task = next((t for t in tasks if t["name"] == task_name), None)
        
        # If no exact match, try case-insensitive match
        if not task:
            task = next((t for t in tasks if t["name"].lower() == task_name.lower()), None)
            
        if task:
            logger.info(f"Found task: {task['name']}")
            task_list.append([task["name"], task["description"]])
        else:
            # If task not found, still add it to the list with a placeholder description
            logger.warning(f"Task not found: {task_name}")
            task_list.append([task_name, "Task not found in saved tasks"])
    
    # Convert to DataFrame with explicit column names
    if task_list:
        logger.info(f"Loaded playlist '{playlist_name}' with {len(task_list)} tasks")
        task_df = pd.DataFrame(task_list, columns=["Task Name", "Description"])
    else:
        logger.warning(f"Loaded playlist '{playlist_name}' but it contains no tasks")
        # Return an empty DataFrame with the correct columns
        task_df = pd.DataFrame(columns=["Task Name", "Description"])
    
    return task_df, selected_playlist["name"], selected_playlist.get("description", ""), f"Loaded playlist: {playlist_name} with {len(task_list)} tasks"

def save_current_playlist_for_ui(playlist_name, description="", task_list=None):
    """
    Save the current playlist in the UI.
    
    Args:
        playlist_name (str): Name of the playlist
        description (str): Optional description of the playlist
        task_list (list or DataFrame): List of tasks in the playlist
        
    Returns:
        tuple: (updated_dropdown, feedback_message, task_list)
    """
    import gradio as gr
    import pandas as pd
    
    if not playlist_name:
        logger.warning("No playlist name provided")
        return gr.update(), "Please provide a playlist name", task_list
    
    # Handle empty task list
    task_names = []
    
    if isinstance(task_list, pd.DataFrame):
        if task_list.empty:
            logger.warning(f"Attempted to save empty playlist: {playlist_name}")
            return gr.update(), "Playlist is empty. Please add tasks first.", task_list
        try:
            # Extract task names from DataFrame using .iloc for positional indexing
            task_names = task_list.iloc[:, 0].tolist() if not task_list.empty else []
            logger.info(f"Saving playlist '{playlist_name}' with {len(task_names)} tasks")
        except Exception as e:
            logger.error(f"Error extracting task names from DataFrame: {str(e)}")
            return gr.update(), f"Error saving playlist: {str(e)}", task_list
    else:
        # For regular list
        try:
            if not task_list:
                logger.warning(f"Attempted to save empty playlist: {playlist_name}")
                return gr.update(), "Playlist is empty. Please add tasks first.", task_list
            task_names = [task[0] for task in task_list if isinstance(task, (list, tuple)) and len(task) > 0]
            logger.info(f"Saving playlist '{playlist_name}' with {len(task_names)} tasks")
        except Exception as e:
            logger.error(f"Error extracting task names from list: {str(e)}")
            return gr.update(), f"Error saving playlist: {str(e)}", task_list
    
    if not task_names:
        logger.warning(f"No tasks to save for playlist: {playlist_name}")
        return gr.update(), "No tasks to save. Please add tasks to the playlist first.", task_list
    
    # Ensure task_names is a simple list of strings for JSON serialization
    task_names = [str(name) for name in task_names]
    
    success = save_playlist(playlist_name, task_names, description)
    
    if success:
        # Refresh the dropdown
        playlists = load_playlists()
        playlist_names = [playlist["name"] for playlist in playlists]
        logger.info(f"Successfully saved playlist '{playlist_name}' with {len(task_names)} tasks")
        
        # Return a completely new dropdown with updated choices
        return gr.Dropdown(choices=playlist_names, value=playlist_name, allow_custom_value=False), f"Playlist '{playlist_name}' saved successfully!", task_list
    else:
        logger.error(f"Failed to save playlist: {playlist_name}")
        return gr.update(), "Failed to save playlist. Please try again.", task_list

def delete_selected_playlist_for_ui(playlist_name):
    """
    Delete a selected playlist from the UI.
    
    Args:
        playlist_name (str): Name of the playlist to delete
        
    Returns:
        tuple: (updated_dropdown, feedback_message)
    """
    import gradio as gr
    
    if not playlist_name:
        return gr.update(), "No playlist selected"
    
    success = delete_playlist(playlist_name)
    
    if success:
        # Refresh the dropdown
        playlists = load_playlists()
        playlist_names = [playlist["name"] for playlist in playlists]
        return gr.update(choices=playlist_names, value=None), f"Playlist '{playlist_name}' deleted successfully!"
    else:
        return gr.update(), "Failed to delete playlist"

def add_task_to_playlist_for_ui(task_name, current_tasks):
    """
    Add a task to the playlist in the UI.
    
    Args:
        task_name (str): Name of the task to add
        current_tasks (list or DataFrame): Current list of tasks in the playlist
        
    Returns:
        tuple: (updated_task_list, feedback_message)
    """
    if not task_name:
        return current_tasks, "No task selected"
    
    from src.utils.utils import load_tasks
    import pandas as pd
    
    tasks = load_tasks()
    task = next((t for t in tasks if t["name"] == task_name), None)
    
    if not task:
        return current_tasks, "Task not found"
    
    # Create the new task entry
    new_task = [task["name"], task["description"]]
    
    # Handle both list and DataFrame objects
    if isinstance(current_tasks, pd.DataFrame):
        # Check if task is already in the playlist
        if not current_tasks.empty and task_name in current_tasks.iloc[:, 0].values:
            return current_tasks, f"Task '{task_name}' is already in the playlist"
            
        # For pandas DataFrame, use concat
        new_row_df = pd.DataFrame([new_task], columns=["Task Name", "Description"])
        updated_tasks = pd.concat([current_tasks, new_row_df], ignore_index=True)
    else:
        # Check if task is already in the playlist for list type
        for existing_task in current_tasks:
            if isinstance(existing_task, (list, tuple)) and len(existing_task) > 0 and existing_task[0] == task_name:
                return current_tasks, f"Task '{task_name}' is already in the playlist"
                
        # For regular list, create a new list with the added task
        updated_tasks = list(current_tasks)  # Create a copy to avoid modifying the original
        updated_tasks.append(new_task)
    
    logger.info(f"Added task '{task_name}' to playlist")
    return updated_tasks, f"Added task '{task_name}' to playlist"

def remove_task_from_playlist_for_ui(task_index, current_tasks):
    """
    Remove a task from the playlist in the UI.
    
    Args:
        task_index (int): Index of the task to remove
        current_tasks (list or DataFrame): Current list of tasks in the playlist
        
    Returns:
        tuple: (updated_task_list, feedback_message)
    """
    import pandas as pd
    
    if task_index < 0 or (isinstance(current_tasks, pd.DataFrame) and task_index >= len(current_tasks)) or (not isinstance(current_tasks, pd.DataFrame) and task_index >= len(current_tasks)):
        return current_tasks, "Invalid task index"
    
    if len(current_tasks) == 0:
        return current_tasks, "No tasks to remove"
    
    # Handle both list and DataFrame objects
    if isinstance(current_tasks, pd.DataFrame):
        # For pandas DataFrame, drop the row
        task_name = current_tasks.iloc[task_index, 0] if len(current_tasks.columns) > 0 else "Unknown"
        updated_tasks = current_tasks.drop(task_index).reset_index(drop=True)
    else:
        # For regular list
        task_name = current_tasks[task_index][0] if len(current_tasks[task_index]) > 0 else "Unknown"
        updated_tasks = current_tasks.copy()
        updated_tasks.pop(task_index)
    
    return updated_tasks, f"Removed task '{task_name}' from playlist"

def move_task_up_for_ui(task_index, current_tasks):
    """
    Move a task up in the playlist in the UI.
    
    Args:
        task_index (int): Index of the task to move
        current_tasks (list or DataFrame): Current list of tasks in the playlist
        
    Returns:
        tuple: (updated_task_list, feedback_message)
    """
    import pandas as pd
    
    if task_index <= 0 or (isinstance(current_tasks, pd.DataFrame) and task_index >= len(current_tasks)) or (not isinstance(current_tasks, pd.DataFrame) and task_index >= len(current_tasks)):
        return current_tasks, "Cannot move task up"
    
    if len(current_tasks) <= 1:
        return current_tasks, "Not enough tasks to reorder"
    
    # Handle both list and DataFrame objects
    if isinstance(current_tasks, pd.DataFrame):
        # For pandas DataFrame
        updated_tasks = current_tasks.copy()
        # Get the task names for feedback message
        task_name = updated_tasks.iloc[task_index, 0] if len(updated_tasks.columns) > 0 else "Unknown"
        # Swap rows
        updated_tasks.iloc[task_index-1], updated_tasks.iloc[task_index] = updated_tasks.iloc[task_index].copy(), updated_tasks.iloc[task_index-1].copy()
    else:
        # For regular list
        updated_tasks = current_tasks.copy()
        task_name = updated_tasks[task_index][0]
        # Swap elements
        updated_tasks[task_index-1], updated_tasks[task_index] = updated_tasks[task_index], updated_tasks[task_index-1]
    
    return updated_tasks, f"Moved task '{task_name}' up"

def move_task_down_for_ui(task_index, current_tasks):
    """
    Move a task down in the playlist in the UI.
    
    Args:
        task_index (int): Index of the task to move
        current_tasks (list or DataFrame): Current list of tasks in the playlist
        
    Returns:
        tuple: (updated_task_list, feedback_message)
    """
    import pandas as pd
    
    if task_index < 0 or task_index >= len(current_tasks) - 1:
        return current_tasks, "Cannot move task down"
    
    if len(current_tasks) <= 1:
        return current_tasks, "Not enough tasks to reorder"
    
    # Handle both list and DataFrame objects
    if isinstance(current_tasks, pd.DataFrame):
        # For pandas DataFrame
        updated_tasks = current_tasks.copy()
        # Get the task name for feedback message
        task_name = updated_tasks.iloc[task_index, 0] if len(updated_tasks.columns) > 0 else "Unknown"
        # Swap rows
        updated_tasks.iloc[task_index], updated_tasks.iloc[task_index+1] = updated_tasks.iloc[task_index+1].copy(), updated_tasks.iloc[task_index].copy()
    else:
        # For regular list
        updated_tasks = current_tasks.copy()
        task_name = updated_tasks[task_index][0]
        # Swap elements
        updated_tasks[task_index], updated_tasks[task_index+1] = updated_tasks[task_index+1], updated_tasks[task_index]
    
    return updated_tasks, f"Moved task '{task_name}' down"

def update_available_tasks_for_ui():
    """
    Update the available tasks dropdown in the UI.
    
    Returns:
        list: List of task names
    """
    import gradio as gr
    from src.utils.utils import load_tasks
    
    tasks = load_tasks()
    task_names = [task["name"] for task in tasks]
    return gr.update(choices=task_names)

async def play_playlist_for_ui(playlist_tasks, agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, 
                              llm_base_url, llm_api_key, use_own_browser, keep_browser_open, headless, disable_security, 
                              window_w, window_h, save_recording_path, save_agent_history_path, save_trace_path, 
                              enable_recording, max_steps, use_vision, max_actions_per_step, tool_calling_method):
    """
    Play a playlist of tasks in the UI.
    
    Args:
        playlist_tasks (list or DataFrame): List of tasks in the playlist
        agent_type (str): Type of agent to use
        llm_provider (str): LLM provider
        llm_model_name (str): LLM model name
        llm_num_ctx (int): LLM context window size
        llm_temperature (float): LLM temperature
        llm_base_url (str): LLM base URL
        llm_api_key (str): LLM API key
        use_own_browser (bool): Whether to use own browser
        keep_browser_open (bool): Whether to keep browser open
        headless (bool): Whether to run in headless mode
        disable_security (bool): Whether to disable security
        window_w (int): Window width
        window_h (int): Window height
        save_recording_path (str): Path to save recording
        save_agent_history_path (str): Path to save agent history
        save_trace_path (str): Path to save trace
        enable_recording (bool): Whether to enable recording
        max_steps (int): Maximum number of steps
        use_vision (bool): Whether to use vision
        max_actions_per_step (int): Maximum number of actions per step
        tool_calling_method (str): Tool calling method
        
    Yields:
        tuple: (current_task_display, play_playlist_button, stop_playlist_button, feedback_message)
    """
    import pandas as pd
    import gradio as gr
    from webui import run_browser_agent
    
    # Check if playlist is empty
    task_list = []
    
    if isinstance(playlist_tasks, pd.DataFrame):
        logger.info(f"Playing playlist with DataFrame type, empty: {playlist_tasks.empty}")
        if playlist_tasks.empty:
            yield gr.update(value="No tasks to play"), gr.update(interactive=True), gr.update(interactive=False), "Playlist is empty"
            return
        # Convert DataFrame to list for consistent processing
        for _, row in playlist_tasks.iterrows():
            if len(row) > 0:
                # Use .iloc for positional indexing to avoid deprecation warning
                task_name = row.iloc[0]  # Get task name from first column
                if task_name:  # Skip empty task names
                    task_list.append(task_name)
    else:
        # Handle list type
        logger.info(f"Playing playlist with list type, length: {len(playlist_tasks) if playlist_tasks else 0}")
        # Fix: Use proper None check and length check separately to avoid ambiguity
        if playlist_tasks is None:
            yield gr.update(value="No tasks to play"), gr.update(interactive=True), gr.update(interactive=False), "Playlist is empty"
            return
        if len(playlist_tasks) == 0:
            yield gr.update(value="No tasks to play"), gr.update(interactive=True), gr.update(interactive=False), "Playlist is empty"
            return
        for task in playlist_tasks:
            if isinstance(task, (list, tuple)) and len(task) > 0:
                task_name = task[0]
                if task_name:  # Skip empty task names
                    task_list.append(task_name)
    
    num_tasks = len(task_list)
    logger.info(f"Playing playlist with {num_tasks} tasks")
    
    if num_tasks == 0:
        yield gr.update(value="No valid tasks to play"), gr.update(interactive=True), gr.update(interactive=False), "Playlist contains no valid tasks"
        return
    
    # Load tasks
    from src.utils.utils import load_tasks
    tasks = load_tasks()
    
    # Iterate over tasks in the playlist
    for task_index, task_name in enumerate(task_list):
        try:
            # Find the task in the loaded tasks
            task = next((t for t in tasks if t["name"] == task_name), None)
            
            if not task:
                yield gr.update(value=f"Task not found: {task_name}"), gr.update(), gr.update(), f"Task not found: {task_name}"
                continue
            
            # Update UI
            current_index = task_index + 1
            yield gr.update(value=f"Playing task {current_index}/{num_tasks}: {task_name}"), gr.update(interactive=False), gr.update(interactive=True), f"Playing task: {task_name}"
            
            # Run the task
            add_infos = task.get("additional_info", "")
            
            # Run the browser agent
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
                task=task["description"],
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
            
            yield gr.update(value=f"Completed task {current_index}/{num_tasks}: {task_name}"), gr.update(), gr.update(), f"Completed task: {task_name}"
            
        except Exception as e:
            logger.error(f"Error playing task: {str(e)}")
            yield gr.update(value=f"Error playing task: {str(e)}"), gr.update(interactive=True), gr.update(interactive=False), f"Error playing task: {str(e)}"
    
    # Playlist completed
    yield gr.update(value="Playlist completed"), gr.update(interactive=True), gr.update(interactive=False), "Playlist completed" 