with open("webui.py", "r") as f: lines = f.readlines(); lines[1410] = "            result = await agent_task
"; lines[1702] = "                with gr.Row():
"; lines[1703] = "                    with gr.Column(scale=2):
"; lines[1704] = "                        task = gr.Textbox(
"; lines[1705] = "                            label=\"Task\",
"; lines[1706] = "                            lines=5,
"; lines[1707] = "                            value=config[\"task\"],
"; lines[1708] = "                            info=\"Describe the task for the agent to perform\",
"; lines[1709] = "                        )
"; lines[1710] = "                        add_infos = gr.Textbox(
"; with open("webui.py", "w") as f: f.writelines(lines)
