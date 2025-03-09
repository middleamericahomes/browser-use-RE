#!/usr/bin/env python3
"""
QWQ-32B Browser Agent Test Script

This script tests the QWQ-32B integration with the browser agent.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from src.utils.utils import get_llm_model
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserConfig, BrowserContextConfig, BrowserContextWindowSize
from browser_use.controller.service import Controller
from src.agent.custom_agent import CustomAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwq-browser-test")

# Load environment variables
load_dotenv()

async def test_qwq_browser_agent():
    """Test the QWQ-32B integration with the browser agent"""
    
    # Initialize browser
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
        )
    )
    
    # Create browser context
    browser_context = await browser.new_context(
        config=BrowserContextConfig(
            no_viewport=False,
            browser_window_size=BrowserContextWindowSize(
                width=1280,
                height=800
            ),
        )
    )
    
    # Initialize controller
    controller = Controller()
    
    # Get QWQ-32B model
    llm = get_llm_model(
        provider="openrouter",
        model_name="qwen/qwq-32b",
        temperature=0.0,
    )
    
    # Create agent
    agent = CustomAgent(
        task="Go to google.com and search for 'OpenAI'. Then tell me the title of the first search result.",
        llm=llm,
        browser=browser,
        browser_context=browser_context,
        controller=controller,
        use_vision=True,
        max_steps=10,
        max_actions_per_step=5,
    )
    
    # Run agent
    try:
        logger.info("Starting QWQ-32B browser agent test...")
        result = await agent.run()
        logger.info(f"Agent completed with result: {result}")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        # Close browser
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_qwq_browser_agent()) 