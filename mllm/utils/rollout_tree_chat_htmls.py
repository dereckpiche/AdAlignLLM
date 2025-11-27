"""
File: mllm/utils/rollout_tree_chat_htmls.py
Summary: Renders rollout tree chat transcripts into HTML artifacts.
"""

from pathlib import Path
from typing import List

from mllm.utils.rollout_tree_gather_utils import *


def html_from_chat_turns(chat_turns: List[ChatTurnLog]) -> str:
    """
    Render chat turns as a single, wrapping sequence of messages in time order.
    Keep badge and message bubble styles, include time on every badge and
    include rewards on assistant badges. Each message is individually
    hide/show by click; when hidden, only the badge remains and "(...)" is
    shown inline (not inside a bubble).
    """
    import html
    import re as _re

    # Prepare ordering: sort by (time_step, original_index) to keep stable order within same step
    indexed_turns = list(enumerate(chat_turns))
    indexed_turns.sort(key=lambda t: (t[1].time_step, t[0]))

    # Get unique agent IDs and sort alphabetically for consistent assignment
    # Agent with alphabetically lower name gets agent-0 (left, green)
    # Agent with alphabetically higher name gets agent-1 (right, orange)
    unique_agent_ids = sorted(
        set(turn.agent_id for turn in chat_turns if turn.role == "assistant")
    )
    agent_id_to_index = {aid: idx for idx, aid in enumerate(unique_agent_ids)}

    # CSS styles (simplified layout; no time-step or agent-column backgrounds)
    css = """
    <style>
        :root {
            --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            --bg: #ffffff;
            --text: #1c0b00;
            --muted-text: #2C3E50;
            --accent-muted: #BDC3C7;
            --accent-muted-2: #D0D7DE;
            --panel-bg: #F8FAFC;
            --reward-color: #3a2e00; /* dark text for reward pill */
            --font-size: 14px;
            --border-width: 2px;
            --corner-radius: 6px;
            --pill-radius-left: 999px 0 0 999px;
            --pill-radius-right: 0 999px 999px 0;
            --inset-shadow: 0 1px 0 rgba(0,0,0,0.03) inset;

            /* Chat View Colors */
            --agent-0-bg: #dcf8c6;
            --agent-0-border: #0eb224;
            --agent-1-bg: #ffe4cc;
            --agent-1-border: #ef8323;
            --user-bg: #f5f5f5;
            --chat-bg: #ffffff;
        }
        body {
            font-family: var(--font-family);
            margin: 12px;
            background-color: var(--bg);
            color: var(--text);
            font-size: var(--font-size);
            line-height: 1.5;
        }

        /* Chat View Styles */
        #flow-chat {
            max-width: 900px;
            margin: 0 auto;
            background: var(--chat-bg);
            padding: 12px 16px 12px 8px;
            border-radius: 8px;
        }

        .simultaneous-messages {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 8px;
            margin-bottom: 4px;
            align-items: flex-start;
            width: 100%;
            overflow: hidden;
            box-sizing: border-box;
        }

        .simultaneous-messages .chat-message {
            flex: 1 1 0 !important;
            margin-bottom: 0 !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: flex-start !important;
            margin-left: 0 !important;
            min-width: 0 !important;
            max-width: 50% !important;
            gap: 0 !important;
            overflow: hidden !important;
        }

        .simultaneous-messages .chat-message-content {
            max-width: 100% !important;
            width: 100%;
            align-items: flex-start !important;
            margin-left: 0 !important;
            overflow: hidden !important;
        }

        .simultaneous-messages .chat-message.agent-0 {
            justify-content: flex-start !important;
        }

        .simultaneous-messages .chat-message.agent-1 {
            justify-content: flex-end !important;
        }

        .simultaneous-messages .chat-message.agent-0 .chat-message-content {
            margin-left: 0 !important;
            align-items: flex-start !important;
        }

        .simultaneous-messages .chat-message.agent-1 .chat-message-content {
            margin-left: auto !important;
            margin-right: 0 !important;
            align-items: flex-end !important;
        }

        .simultaneous-messages .chat-bubble {
            max-width: 100%;
            word-break: break-word;
            overflow-wrap: break-word;
            box-sizing: border-box;
        }

        .simultaneous-messages .chat-message.agent-0 .chat-bubble {
            border-radius: 10px;
        }

        .simultaneous-messages .chat-message.agent-1 .chat-bubble {
            border-radius: 10px;
        }

        .simultaneous-messages .chat-message.agent-0 .chat-header {
            justify-content: flex-start;
            flex-shrink: 0;
        }

        .simultaneous-messages .chat-message.agent-1 .chat-header {
            justify-content: flex-end;
            flex-shrink: 0;
        }

        .simultaneous-messages .chat-reasoning {
            max-width: 100%;
            overflow-wrap: break-word;
        }

        /* Styling for user prompts in simultaneous-messages */
        .simultaneous-messages .chat-message.role-user {
            flex: 1 1 0 !important;
            margin-bottom: 0 !important;
            display: flex !important;
            opacity: 0.7;
            cursor: pointer;
        }

        .simultaneous-messages .chat-message.role-user:hover {
            opacity: 1;
        }

        .simultaneous-messages .chat-message.role-user.collapsed .chat-bubble {
            display: none;
        }

        .simultaneous-messages .chat-message.role-user.collapsed .chat-header::after {
            content: ' (collapsed)';
            font-weight: normal;
            font-style: italic;
            color: #999;
            font-size: 0.9em;
        }

        .simultaneous-messages .chat-message.role-user.agent-0 {
            justify-content: flex-start !important;
        }

        .simultaneous-messages .chat-message.role-user.agent-1 {
            justify-content: flex-end !important;
        }

        .simultaneous-messages .chat-message.role-user.agent-0 .chat-message-content {
            margin-left: 0 !important;
            align-items: flex-start !important;
        }

        .simultaneous-messages .chat-message.role-user.agent-1 .chat-message-content {
            margin-left: auto !important;
            margin-right: 0 !important;
            align-items: flex-end !important;
        }

        /* Styling for split-agent-context when wrapped */
        .simultaneous-messages .split-agent-context {
            width: 100%;
            display: flex !important;
        }

        .chat-message {
            display: flex;
            margin-bottom: 2px;
            align-items: flex-end;
            gap: 6px;
            position: relative;
            margin-left: 36px;
        }

        .chat-message.agent-0 {
            margin-left: 0;
        }

        .chat-message.agent-1 {
            margin-left: 0;
        }

        .chat-message.agent-0::before {
            left: 0;
        }

        .chat-message.agent-1::before {
            left: 0;
        }

        .chat-message.role-user {
            opacity: 0.7;
            cursor: pointer;
        }

        .chat-message.role-user.collapsed .chat-bubble {
            display: none;
        }

        .chat-message.role-user.collapsed .chat-header::after {
            content: ' (collapsed)';
            font-weight: normal;
            font-style: italic;
            color: #999;
            font-size: 0.9em;
        }

        .chat-message.role-user:hover {
            opacity: 1;
        }

        .chat-message::before {
            content: '';
            position: absolute;
            left: -36px;
            top: 0;
            bottom: 0;
            width: 36px;
            pointer-events: auto;
        }

        .merge-btn {
            position: absolute;
            left: -30px;
            top: 50%;
            transform: translateY(-50%);
            width: 26px;
            height: 26px;
            border-radius: 4px;
            border: 1.5px solid var(--accent-muted);
            background: white;
            cursor: pointer;
            font-size: var(--font-size);
            opacity: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: opacity 0.2s ease, transform 0.1s ease;
            padding: 0;
            line-height: 1;
            z-index: 10;
        }

        .chat-message:hover .merge-btn,
        .merge-btn:hover {
            opacity: 1;
        }

        .merge-btn:hover {
            background: var(--panel-bg);
            border-color: var(--accent-muted-2);
            transform: translateY(-50%) scale(1.15);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }

        .merge-btn:active {
            transform: translateY(-50%) scale(0.95);
        }

        .chat-message.agent-0 .merge-btn {
            left: -30px;
        }

        .chat-message.agent-1 .merge-btn {
            left: -30px;
        }

        .chat-message.role-user .merge-btn {
            display: none !important;
        }

        .simultaneous-messages .merge-btn {
            opacity: 0 !important;
            pointer-events: none;
        }

        .simultaneous-messages {
            padding: 6px 0 6px 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            position: relative !important;
            background: transparent !important;
            border-radius: 0 !important;
            box-sizing: border-box !important;
            overflow: visible !important;
            max-width: 100% !important;
            border: none !important;
            transition: padding 0.2s ease !important;
        }

        .simultaneous-messages:hover {
            padding-top: 40px !important;
        }

        .simultaneous-messages::before {
            content: '⇅ Merged';
            position: absolute;
            left: 0 !important;
            top: 8px !important;
            font-size: var(--font-size);
            font-weight: 500;
            color: #888;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease;
        }

        .simultaneous-messages:hover::before {
            opacity: 1;
        }

        .unmerge-btn {
            position: absolute !important;
            right: 0 !important;
            top: 6px !important;
            width: 36px !important;
            height: 28px !important;
            border-radius: 5px !important;
            border: 2px solid #d63031 !important;
            background: white !important;
            cursor: pointer !important;
            font-size: var(--font-size) !important;
            font-weight: bold !important;
            color: #d63031 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s ease !important;
            padding: 0 !important;
            line-height: 1 !important;
            z-index: 1000 !important;
            flex: none !important;
            pointer-events: auto !important;
            box-shadow: 0 2px 6px rgba(214, 48, 49, 0.3) !important;
            opacity: 0 !important;
        }

        .simultaneous-messages:hover .unmerge-btn {
            opacity: 1 !important;
        }

        .unmerge-btn:hover {
            background: #ffe5e5 !important;
            border-color: #b71c1c !important;
            transform: scale(1.1) !important;
            box-shadow: 0 3px 8px rgba(214, 48, 49, 0.4) !important;
        }

        .unmerge-btn:active {
            transform: scale(0.95) !important;
            background: #ffcccc !important;
        }

        .chat-message-content {
            max-width: 72%;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .chat-message.agent-0 .chat-message-content {
            align-items: flex-start;
        }

        .chat-message.agent-1 .chat-message-content {
            align-items: flex-end;
            margin-left: auto;
        }

        .chat-bubble {
            padding: 6px 10px;
            border-radius: 10px;
            word-wrap: break-word;
            position: relative;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            line-height: 1.4;
        }

        .chat-message.agent-0 .chat-bubble {
            background: var(--agent-0-bg);
            border: 2px solid var(--agent-0-border);
            border-radius: 10px 10px 10px 2px;
        }

        .chat-message.agent-1 .chat-bubble {
            background: var(--agent-1-bg);
            border: 2px solid var(--agent-1-border);
            border-radius: 10px 10px 2px 10px;
        }

        .chat-message.role-user .chat-bubble {
            background: var(--user-bg);
            border: 2px solid #d0d0d0;
        }

        .chat-header {
            display: flex;
            align-items: center;
            gap: 4px;
            margin-bottom: 2px;
            font-size: var(--font-size);
            font-weight: 600;
            line-height: 1.2;
        }

        .chat-message.agent-0 .chat-header {
            color: var(--agent-0-border);
        }

        .chat-message.agent-1 .chat-header {
            color: var(--agent-1-border);
        }

        .chat-timestamp {
            font-size: var(--font-size);
            color: var(--muted-text);
            margin-top: 1px;
            opacity: 0.75;
        }

        .chat-reward {
            display: inline-flex;
            align-items: center;
            background: linear-gradient(90deg, #fffdf2 0%, #ffffff 75%);
            color: #000000;
            font-weight: 600;
            font-size: var(--font-size);
            padding: 1px 5px;
            border-radius: 3px;
            border: 1px solid #f4e6a8;
            margin-left: 4px;
            line-height: 1.3;
        }

        .chat-reasoning {
            font-size: var(--font-size);
            font-style: italic;
            color: #555;
            margin-bottom: 2px;
            padding: 4px 8px;
            background: rgba(0, 0, 0, 0.03);
            border-radius: 5px;
            cursor: pointer;
            line-height: 1.3;
        }

        .chat-reasoning.collapsed .reasoning-text {
            display: none;
        }

        .chat-reasoning.collapsed::after {
            content: ' (click to expand)';
            color: #777;
        }

        .chat-group-divider {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            margin: 8px 0 4px 0;
            position: relative;
            cursor: pointer;
            user-select: none;
        }

        .chat-group-divider::before,
        .chat-group-divider::after {
            content: "";
            flex: 1 1 auto;
            height: 2px;
            background: linear-gradient(90deg, rgba(224,230,235,0), var(--accent-muted-2) 30%, var(--accent-muted-2) 70%, rgba(224,230,235,0));
        }

        .chat-group-label {
            display: inline-block;
            background: white;
            padding: 2px 12px;
            border-radius: 999px;
            font-size: var(--font-size);
            font-weight: 700;
            color: var(--muted-text);
            border: 1.5px solid var(--accent-muted);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            line-height: 1.4;
            position: relative;
            transition: background 0.2s ease;
        }

        .chat-group-divider:hover .chat-group-label {
            background: var(--panel-bg);
        }

        .chat-group-label::before {
            content: '▼ ';
            font-size: 0.8em;
            display: inline-block;
            transition: transform 0.2s ease;
            opacity: 0;
        }

        .chat-group-divider:hover .chat-group-label::before {
            opacity: 1;
        }

        .chat-group-divider.collapsed .chat-group-label::before {
            content: '▶ ';
            opacity: 1;
        }

        .chat-group-divider.collapsed + * {
            display: none !important;
        }

        /* Hide collapsed rounds in strong hide mode */
        .strong-hide .chat-group-divider.collapsed {
            display: none !important;
        }

        /* Chat view width control */
        #flow-chat {
            --chat-width: 900px;
            max-width: var(--chat-width);
            margin: 0 auto;
        }

        /* Hide user messages when toggle is on */
        #flow-chat.hide-user-messages .chat-message.role-user {
            display: none;
        }

        /* Hide rewards when hiding user messages */
        #flow-chat.hide-user-messages .chat-reward {
            display: none;
        }

        /* Round context annotations */
        .round-context {
            text-align: center;
            margin: 4px auto;
            max-width: 100%;
        }

        .round-context-edit {
            min-height: 20px;
            padding: 5px 10px;
            border: 1.5px dashed var(--accent-muted);
            border-radius: 6px;
            background: #fafafa;
            cursor: text;
            transition: all 0.2s ease;
            outline: none;
            font-size: var(--font-size);
            line-height: 1.3;
            user-select: text;
            -webkit-user-select: text;
            -moz-user-select: text;
            -ms-user-select: text;
        }

        .round-context-edit:focus {
            border-style: solid;
            border-color: var(--accent-muted-2);
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .round-context-edit:empty:before {
            content: attr(data-placeholder);
            color: #999;
            font-style: italic;
        }

        .round-context-controls {
            display: none;
            justify-content: center;
            gap: 4px;
            margin-top: 4px;
            flex-wrap: wrap;
        }

        .round-context-edit:focus + .round-context-controls,
        .round-context-controls:hover,
        .round-context:focus-within .round-context-controls {
            display: flex;
        }

        .context-color-btn {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 1.5px solid #fff;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
            cursor: pointer;
            transition: transform 0.1s ease;
        }

        .context-color-btn:hover {
            transform: scale(1.15);
        }

        .context-color-btn:active {
            transform: scale(0.95);
        }

        /* Split agent context boxes */
        .split-agent-context {
            display: flex;
            gap: 6px;
            margin: 4px auto;
            max-width: 100%;
            align-items: flex-start;
        }

        .agent-context-box {
            flex: 1;
            min-width: 0;
            position: relative;
        }

        .agent-context-box .round-context-edit {
            margin: 0;
            border-radius: 6px;
            padding: 4px 8px;
            min-height: 18px;
        }

        .agent-context-box.agent-0 .round-context-edit {
            border-color: var(--agent-0-border);
            background: rgba(14, 178, 36, 0.03);
        }

        .agent-context-box.agent-1 .round-context-edit {
            border-color: var(--agent-1-border);
            background: rgba(239, 131, 35, 0.03);
        }

        .agent-context-box.agent-0 .round-context-edit:focus {
            border-color: var(--agent-0-border);
            box-shadow: 0 2px 8px rgba(14, 178, 36, 0.2);
            background: rgba(14, 178, 36, 0.05);
        }

        .agent-context-box.agent-1 .round-context-edit:focus {
            border-color: var(--agent-1-border);
            box-shadow: 0 2px 8px rgba(239, 131, 35, 0.2);
            background: rgba(239, 131, 35, 0.05);
        }

        .agent-context-box .round-context-edit::before {
            font-weight: 700;
            font-size: var(--font-size);
            margin-right: 5px;
            letter-spacing: 0.2px;
        }

        .agent-context-box.agent-0 .round-context-edit::before {
            content: 'Agent 0 Prompt Summary:';
            color: var(--agent-0-border);
        }

        .agent-context-box.agent-1 .round-context-edit::before {
            content: 'Agent 1 Prompt Summary:';
            color: var(--agent-1-border);
        }

        /* Empty context boxes will be hidden by JavaScript when strong hide is enabled */
        .toolbar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0;
            font-size: var(--font-size);
            max-height: 0;
            overflow: hidden;
            opacity: 0;
            pointer-events: none;
            transition: max-height 0.2s ease, opacity 0.2s ease;
            flex-wrap: wrap;
        }
        .toolbar-wrap { position: sticky; top: 0; z-index: 10; background: var(--bg); }
        .toolbar-hotzone { height: 6px; }
        .toolbar-wrap:hover .toolbar { max-height: 500px; opacity: 1; pointer-events: auto; margin-bottom: 12px; }
        .toolbar * { pointer-events: auto !important; }
        .toolbar input,
        .toolbar select { z-index: 100 !important; position: relative; }
        .toolbar input[type="number"],
        .toolbar input[type="text"],
        .toolbar select {
            width: 72px;
            padding: 2px 6px;
            border: 1px solid var(--accent-muted);
            border-radius: var(--corner-radius);
            background: var(--bg);
            user-select: text !important;
            -webkit-user-select: text !important;
            -moz-user-select: text !important;
            -ms-user-select: text !important;
            pointer-events: auto !important;
            cursor: pointer !important;
        }
        .toolbar input[type="text"] {
            cursor: text !important;
        }
        .toolbar input[type="text"]:focus,
        .toolbar input[type="number"]:focus,
        .toolbar select:focus {
            outline: 2px solid #0066cc;
            outline-offset: 1px;
        }
        .toolbar button {
            padding: 4px 8px;
            border: 1px solid var(--accent-muted);
            background: var(--panel-bg);
            border-radius: var(--corner-radius);
            cursor: pointer;
        }
        .emoji-bw { filter: grayscale(100%); opacity: 0.95; font-size: var(--font-size); vertical-align: baseline; margin: 0; position: relative; top: -1px; line-height: 1; display: inline-block; }
    </style>
    """

    # HTML structure
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>Chat Turns</title>",
        css,
        "<script>\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "  const chatFlow = document.getElementById('flow-chat');\n"
        "  let strongHideOn = false;\n"
        "  let hideUserMessages = false;\n"
        "  const hideUserBtn = document.getElementById('toggle-hide-user-messages');\n"
        "  const hideUserStateEl = document.getElementById('hide-user-state');\n"
        "  const widthControl = document.getElementById('chat-width-control');\n"
        "  const widthSlider = document.getElementById('chat-width-slider');\n"
        "  const widthValue = document.getElementById('chat-width-value');\n"
        "  const strongHideBtn = document.getElementById('toggle-strong-hide');\n"
        "  const strongHideStateEl = document.getElementById('strong-hide-state');\n"
        "  if (strongHideBtn) {\n"
        "    const setLabel = () => { if (strongHideStateEl) { strongHideStateEl.textContent = strongHideOn ? 'On' : 'Off'; } };\n"
        "    strongHideBtn.addEventListener('click', () => { strongHideOn = !strongHideOn; chatFlow.classList.toggle('strong-hide', strongHideOn); setLabel(); applyStrongHideToChat(); });\n"
        "    setLabel();\n"
        "  }\n"
        "  if (hideUserBtn && hideUserStateEl && chatFlow) {\n"
        "    const updateHideUser = () => { hideUserStateEl.textContent = hideUserMessages ? 'On' : 'Off'; };\n"
        "    hideUserBtn.addEventListener('click', () => {\n"
        "      hideUserMessages = !hideUserMessages;\n"
        "      chatFlow.classList.toggle('hide-user-messages', hideUserMessages);\n"
        "      updateHideUser();\n"
        "    });\n"
        "    updateHideUser();\n"
        "  }\n"
        "  if (widthSlider && widthValue && chatFlow) {\n"
        "    const savedWidth = localStorage.getItem('chat-view-width');\n"
        "    if (savedWidth) {\n"
        "      widthSlider.value = savedWidth;\n"
        "      chatFlow.style.setProperty('--chat-width', savedWidth + 'px');\n"
        "      widthValue.textContent = savedWidth + 'px';\n"
        "    }\n"
        "    widthSlider.addEventListener('input', (e) => {\n"
        "      const width = e.target.value;\n"
        "      chatFlow.style.setProperty('--chat-width', width + 'px');\n"
        "      widthValue.textContent = width + 'px';\n"
        "      localStorage.setItem('chat-view-width', width);\n"
        "    });\n"
        "  }\n"
        "  const fontFamilySelect = document.getElementById('font-family-select');\n"
        "  const fontSizeInput = document.getElementById('font-size-input');\n"
        "  if (fontFamilySelect) {\n"
        "    const savedFont = localStorage.getItem('render-font-family');\n"
        "    if (savedFont) {\n"
        "      fontFamilySelect.value = savedFont;\n"
        "      document.body.style.setProperty('--font-family', savedFont);\n"
        "    }\n"
        "    fontFamilySelect.addEventListener('change', (e) => {\n"
        "      const font = e.target.value;\n"
        "      document.body.style.setProperty('--font-family', font);\n"
        "      localStorage.setItem('render-font-family', font);\n"
        "    });\n"
        "  }\n"
        "  if (fontSizeInput) {\n"
        "    const savedSize = localStorage.getItem('render-font-size');\n"
        "    if (savedSize) {\n"
        "      fontSizeInput.value = savedSize;\n"
        "      document.body.style.setProperty('--font-size', savedSize + 'px');\n"
        "    }\n"
        "    fontSizeInput.addEventListener('input', (e) => {\n"
        "      const size = e.target.value;\n"
        "      document.body.style.setProperty('--font-size', size + 'px');\n"
        "      localStorage.setItem('render-font-size', size);\n"
        "    });\n"
        "  }\n"
        "  const agent0EmojiInput = document.getElementById('agent0-emoji-input');\n"
        "  const agent0NameInput = document.getElementById('agent0-name-input');\n"
        "  const agent1EmojiInput = document.getElementById('agent1-emoji-input');\n"
        "  const agent1NameInput = document.getElementById('agent1-name-input');\n"
        "  const applyAgentNamesBtn = document.getElementById('apply-agent-names');\n"
        "  function loadAgentNames() {\n"
        "    if (agent0EmojiInput && agent0NameInput && agent1EmojiInput && agent1NameInput) {\n"
        "      const savedAgent0Emoji = localStorage.getItem('agent0-emoji') || '🤖';\n"
        "      const savedAgent0Name = localStorage.getItem('agent0-name') || document.getElementById('agent0-name-input').placeholder;\n"
        "      const savedAgent1Emoji = localStorage.getItem('agent1-emoji') || '🤖';\n"
        "      const savedAgent1Name = localStorage.getItem('agent1-name') || document.getElementById('agent1-name-input').placeholder;\n"
        "      agent0EmojiInput.value = savedAgent0Emoji;\n"
        "      agent0NameInput.value = savedAgent0Name;\n"
        "      agent1EmojiInput.value = savedAgent1Emoji;\n"
        "      agent1NameInput.value = savedAgent1Name;\n"
        "      applyAgentNamesToDOM(savedAgent0Emoji, savedAgent0Name, savedAgent1Emoji, savedAgent1Name);\n"
        "    }\n"
        "  }\n"
        "  function applyAgentNamesToDOM(agent0Emoji, agent0Name, agent1Emoji, agent1Name) {\n"
        "    const agentMap = { '0': { name: agent0Name, emoji: agent0Emoji }, '1': { name: agent1Name, emoji: agent1Emoji } };\n"
        "    document.querySelectorAll('[data-agent-index]').forEach(el => {\n"
        "      const agentIndex = el.getAttribute('data-agent-index');\n"
        "      if (!agentMap[agentIndex]) return;\n"
        "      if (el.classList.contains('agent-name')) {\n"
        "        el.textContent = agentMap[agentIndex].name;\n"
        "      } else if (el.classList.contains('emoji-bw')) {\n"
        "        const currentEmoji = el.textContent.trim();\n"
        "        if (currentEmoji === '🤖' || currentEmoji === '👤') {\n"
        "          el.textContent = agentMap[agentIndex].emoji;\n"
        "        }\n"
        "      }\n"
        "    });\n"
        "    const style = document.createElement('style');\n"
        "    style.id = 'dynamic-agent-names-style';\n"
        "    const existingStyle = document.getElementById('dynamic-agent-names-style');\n"
        "    if (existingStyle) existingStyle.remove();\n"
        "    style.textContent = `\n"
        "      .agent-context-box.agent-0 .round-context-edit::before {\n"
        "        content: '${agent0Name} Prompt Summary:';\n"
        "      }\n"
        "      .agent-context-box.agent-1 .round-context-edit::before {\n"
        "        content: '${agent1Name} Prompt Summary:';\n"
        "      }\n"
        "    `;\n"
        "    document.head.appendChild(style);\n"
        "  }\n"
        "  if (applyAgentNamesBtn && agent0EmojiInput && agent0NameInput && agent1EmojiInput && agent1NameInput) {\n"
        "    [agent0EmojiInput, agent0NameInput, agent1EmojiInput, agent1NameInput].forEach(input => {\n"
        "      input.style.pointerEvents = 'auto';\n"
        "      if (input.tagName === 'INPUT') {\n"
        "        input.style.userSelect = 'text';\n"
        "        input.style.webkitUserSelect = 'text';\n"
        "        input.readOnly = false;\n"
        "      }\n"
        "      input.disabled = false;\n"
        "      const stopAll = (e) => { e.stopPropagation(); e.stopImmediatePropagation(); };\n"
        "      input.addEventListener('mousedown', stopAll, true);\n"
        "      input.addEventListener('mouseup', stopAll, true);\n"
        "      input.addEventListener('click', stopAll, true);\n"
        "      input.addEventListener('dblclick', stopAll, true);\n"
        "      input.addEventListener('focus', stopAll, true);\n"
        "      input.addEventListener('blur', stopAll, true);\n"
        "      input.addEventListener('paste', stopAll, true);\n"
        "      input.addEventListener('cut', stopAll, true);\n"
        "      input.addEventListener('copy', stopAll, true);\n"
        "      input.addEventListener('select', stopAll, true);\n"
        "      input.addEventListener('selectstart', stopAll, true);\n"
        "      input.addEventListener('keydown', stopAll, true);\n"
        "      input.addEventListener('keyup', stopAll, true);\n"
        "      input.addEventListener('keypress', stopAll, true);\n"
        "      input.addEventListener('input', stopAll, true);\n"
        "      input.addEventListener('change', stopAll, true);\n"
        "      input.addEventListener('contextmenu', stopAll, true);\n"
        "    });\n"
        "    const applyNames = () => {\n"
        "      const agent0Emoji = agent0EmojiInput.value || '🤖';\n"
        "      const agent0Name = agent0NameInput.value.trim() || agent0NameInput.placeholder;\n"
        "      const agent1Emoji = agent1EmojiInput.value || '🤖';\n"
        "      const agent1Name = agent1NameInput.value.trim() || agent1NameInput.placeholder;\n"
        "      localStorage.setItem('agent0-emoji', agent0Emoji);\n"
        "      localStorage.setItem('agent0-name', agent0Name);\n"
        "      localStorage.setItem('agent1-emoji', agent1Emoji);\n"
        "      localStorage.setItem('agent1-name', agent1Name);\n"
        "      applyAgentNamesToDOM(agent0Emoji, agent0Name, agent1Emoji, agent1Name);\n"
        "    };\n"
        "    applyAgentNamesBtn.addEventListener('click', applyNames);\n"
        "    [agent0NameInput, agent1NameInput].forEach(input => {\n"
        "      input.addEventListener('keydown', (e) => {\n"
        "        if (e.key === 'Enter') {\n"
        "          e.preventDefault();\n"
        "          e.stopPropagation();\n"
        "          e.stopImmediatePropagation();\n"
        "          applyNames();\n"
        "        }\n"
        "      }, true);\n"
        "    });\n"
        "    [agent0EmojiInput, agent1EmojiInput].forEach(select => {\n"
        "      select.addEventListener('change', applyNames);\n"
        "    });\n"
        "  }\n"
        "  loadAgentNames();\n"
        "  function setupRoundCollapse() {\n"
        "    document.addEventListener('click', function(e) {\n"
        "      if (e.target.closest('input, textarea, select, button, .round-context-edit, .toolbar')) { return; }\n"
        "      const divider = e.target.closest('.chat-group-divider, .group-divider');\n"
        "      if (!divider) return;\n"
        "      divider.classList.toggle('collapsed');\n"
        "      const isCollapsed = divider.classList.contains('collapsed');\n"
        "      let nextElement = divider.nextElementSibling;\n"
        "      while (nextElement) {\n"
        "        if (nextElement.classList.contains('chat-group-divider') || nextElement.classList.contains('group-divider')) {\n"
        "          break;\n"
        "        }\n"
        "        if (isCollapsed) {\n"
        "          if (!nextElement.dataset.originalDisplay) {\n"
        "            nextElement.dataset.originalDisplay = nextElement.style.display || getComputedStyle(nextElement).display;\n"
        "          }\n"
        "          nextElement.style.display = 'none';\n"
        "        } else {\n"
        "          if (nextElement.dataset.originalDisplay) {\n"
        "            const originalDisplay = nextElement.dataset.originalDisplay;\n"
        "            nextElement.style.display = originalDisplay === 'none' ? '' : originalDisplay;\n"
        "            if (nextElement.style.display === originalDisplay && originalDisplay !== 'none') {\n"
        "              nextElement.style.display = '';\n"
        "            }\n"
        "            delete nextElement.dataset.originalDisplay;\n"
        "          } else {\n"
        "            nextElement.style.display = '';\n"
        "          }\n"
        "        }\n"
        "        nextElement = nextElement.nextElementSibling;\n"
        "      }\n"
        "      e.stopPropagation();\n"
        "    });\n"
        "  }\n"
        "  setupRoundCollapse();\n"
        "  const strongHideBtnChat = document.getElementById('toggle-strong-hide');\n"
        "  function applyStrongHideToChat() {\n"
        "    if (!chatFlow) return;\n"
        "    chatFlow.classList.toggle('strong-hide', strongHideOn);\n"
        "    const contextEdits = chatFlow.querySelectorAll('.round-context-edit');\n"
        "    contextEdits.forEach(edit => {\n"
        "      const parent = edit.closest('.round-context, .agent-context-box, .split-agent-context');\n"
        "      if (parent) {\n"
        "        if (strongHideOn && edit.textContent.trim() === '') {\n"
        "          parent.style.display = 'none';\n"
        "        } else {\n"
        "          parent.style.display = '';\n"
        "        }\n"
        "      }\n"
        "    });\n"
        "    const splitContexts = chatFlow.querySelectorAll('.split-agent-context');\n"
        "    splitContexts.forEach(split => {\n"
        "      if (strongHideOn) {\n"
        "        const boxes = split.querySelectorAll('.agent-context-box');\n"
        "        const allEmpty = Array.from(boxes).every(box => {\n"
        "          const edit = box.querySelector('.round-context-edit');\n"
        "          return edit && edit.textContent.trim() === '';\n"
        "        });\n"
        "        if (allEmpty) split.style.display = 'none';\n"
        "      }\n"
        "    });\n"
        "  }\n"
        "  if (strongHideBtnChat && chatFlow) {\n"
        "    strongHideBtnChat.addEventListener('click', () => {\n"
        "      setTimeout(() => applyStrongHideToChat(), 0);\n"
        "    });\n"
        "  }\n"
        "  document.addEventListener('click', function(e) {\n"
        "    if (e.target.closest('input, textarea, select, .round-context-edit, .toolbar')) { return; }\n"
        "    const chatReasoning = e.target.closest('.chat-reasoning');\n"
        "    if (chatReasoning) {\n"
        "      chatReasoning.classList.toggle('collapsed');\n"
        "      return;\n"
        "    }\n"
        "    const userMessage = e.target.closest('.chat-message.role-user');\n"
        "    if (userMessage && !e.target.closest('.merge-btn, .unmerge-btn')) {\n"
        "      userMessage.classList.toggle('collapsed');\n"
        "    }\n"
        "  });\n"
        "  function applyColorToSelection(color, element) {\n"
        "    const selection = window.getSelection();\n"
        "    if (!selection.rangeCount) return false;\n"
        "    const range = selection.getRangeAt(0);\n"
        "    if (!element.contains(range.commonAncestorContainer)) return false;\n"
        "    const selectedText = range.toString();\n"
        "    if (!selectedText) return false;\n"
        "    if (color === 'default') {\n"
        "      // Remove styling - just extract the text content\n"
        "      const textNode = document.createTextNode(selectedText);\n"
        "      range.deleteContents();\n"
        "      range.insertNode(textNode);\n"
        "    } else {\n"
        "      const span = document.createElement('span');\n"
        "      span.style.color = color;\n"
        "      span.style.fontWeight = '600';\n"
        "      try {\n"
        "        range.surroundContents(span);\n"
        "      } catch (e) {\n"
        "        const contents = range.extractContents();\n"
        "        span.appendChild(contents);\n"
        "        range.insertNode(span);\n"
        "      }\n"
        "    }\n"
        "    return true;\n"
        "  }\n"
        "  let lastFocusedContextEdit = null;\n"
        "  document.addEventListener('focusin', function(e) {\n"
        "    if (e.target.classList.contains('round-context-edit')) {\n"
        "      lastFocusedContextEdit = e.target;\n"
        "    }\n"
        "  });\n"
        "  document.addEventListener('mousedown', function(e) {\n"
        "    if (e.target.classList.contains('context-color-btn')) {\n"
        "      e.preventDefault();\n"
        "    }\n"
        "  });\n"
        "  document.addEventListener('click', function(e) {\n"
        "    if (e.target.closest('input:not(.round-context-edit), textarea, select') && !e.target.classList.contains('context-color-btn')) { return; }\n"
        "    if (e.target.classList.contains('context-color-btn')) {\n"
        "      e.preventDefault();\n"
        "      const color = e.target.dataset.color;\n"
        "      const controls = e.target.closest('.round-context-controls');\n"
        "      const contextEdit = controls ? controls.previousElementSibling : null;\n"
        "      if (contextEdit && contextEdit.classList.contains('round-context-edit')) {\n"
        "        contextEdit.focus();\n"
        "        const selection = window.getSelection();\n"
        "        if (selection.rangeCount > 0 && selection.toString().length > 0 && contextEdit.contains(selection.anchorNode)) {\n"
        "          if (applyColorToSelection(color, contextEdit)) {\n"
        "            const key = contextEdit.dataset.contextKey;\n"
        "            localStorage.setItem(key, contextEdit.innerHTML);\n"
        "          }\n"
        "        } else {\n"
        "          try {\n"
        "            if (color !== 'default') {\n"
        "              document.execCommand('styleWithCSS', false, true);\n"
        "              document.execCommand('foreColor', false, color);\n"
        "            }\n"
        "            const key = contextEdit.dataset.contextKey;\n"
        "            setTimeout(() => localStorage.setItem(key, contextEdit.innerHTML), 10);\n"
        "          } catch (e) {\n"
        "            console.log('Color command failed:', e);\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  });\n"
        "  const contextEdits = document.querySelectorAll('.round-context-edit');\n"
        "  contextEdits.forEach(edit => {\n"
        "    edit.addEventListener('input', function() {\n"
        "      const key = this.dataset.contextKey;\n"
        "      localStorage.setItem(key, this.innerHTML);\n"
        "    });\n"
        "    const key = edit.dataset.contextKey;\n"
        "    const saved = localStorage.getItem(key);\n"
        "    if (saved) {\n"
        "      edit.innerHTML = saved;\n"
        "    }\n"
        "  });\n"
        "  document.addEventListener('click', function(e) {\n"
        "    if (e.target.closest('input, textarea, select, .round-context-edit') && !e.target.classList.contains('merge-btn') && !e.target.classList.contains('unmerge-btn')) { return; }\n"
        "    if (e.target.classList.contains('merge-btn')) {\n"
        "      e.preventDefault();\n"
        "      e.stopPropagation();\n"
        "      const msgId = e.target.dataset.msgId;\n"
        "      const currentMsg = e.target.closest('.chat-message');\n"
        "      if (!currentMsg) return;\n"
        "      if (currentMsg.classList.contains('role-user')) {\n"
        "        alert('Cannot merge user messages');\n"
        "        return;\n"
        "      }\n"
        "      let nextMsg = currentMsg.nextElementSibling;\n"
        "      while (nextMsg && !nextMsg.classList.contains('chat-message')) {\n"
        "        nextMsg = nextMsg.nextElementSibling;\n"
        "      }\n"
        "      while (nextMsg && nextMsg.classList.contains('role-user')) {\n"
        "        nextMsg = nextMsg.nextElementSibling;\n"
        "        while (nextMsg && !nextMsg.classList.contains('chat-message')) {\n"
        "          nextMsg = nextMsg.nextElementSibling;\n"
        "        }\n"
        "      }\n"
        "      if (!nextMsg || nextMsg.classList.contains('chat-message') === false) {\n"
        "        alert('No next assistant message to merge with');\n"
        "        return;\n"
        "      }\n"
        "      if (nextMsg.classList.contains('role-user')) {\n"
        "        alert('Cannot merge with user messages');\n"
        "        return;\n"
        "      }\n"
        "      \n"
        "      // Find the user prompts that precede each assistant message\n"
        "      let currentPrompt = currentMsg.previousElementSibling;\n"
        "      while (currentPrompt && !currentPrompt.classList.contains('chat-message')) {\n"
        "        currentPrompt = currentPrompt.previousElementSibling;\n"
        "      }\n"
        "      if (currentPrompt && !currentPrompt.classList.contains('role-user')) {\n"
        "        currentPrompt = null;\n"
        "      }\n"
        "      \n"
        "      let nextPrompt = nextMsg.previousElementSibling;\n"
        "      while (nextPrompt && !nextPrompt.classList.contains('chat-message')) {\n"
        "        nextPrompt = nextPrompt.previousElementSibling;\n"
        "      }\n"
        "      if (nextPrompt && !nextPrompt.classList.contains('role-user')) {\n"
        "        nextPrompt = null;\n"
        "      }\n"
        "      \n"
        "      // Find the split-agent-context that precedes the first prompt or assistant message\n"
        "      let splitContext = null;\n"
        "      let searchStart = currentPrompt || currentMsg;\n"
        "      let elem = searchStart.previousElementSibling;\n"
        "      while (elem) {\n"
        "        if (elem.classList.contains('split-agent-context')) {\n"
        "          splitContext = elem;\n"
        "          break;\n"
        "        }\n"
        "        if (elem.classList.contains('chat-message') || elem.classList.contains('chat-group-divider')) {\n"
        "          break;\n"
        "        }\n"
        "        elem = elem.previousElementSibling;\n"
        "      }\n"
        "      \n"
        "      const parent = currentMsg.parentElement;\n"
        "      if (parent.classList.contains('simultaneous-messages')) {\n"
        "        const wrapper = parent;\n"
        "        currentMsg.style.display = '';\n"
        "        currentMsg.classList.remove('merged');\n"
        "        const refNode = wrapper.nextElementSibling;\n"
        "        parent.parentElement.insertBefore(currentMsg, refNode);\n"
        "        if (nextMsg.parentElement === wrapper) {\n"
        "          parent.parentElement.insertBefore(nextMsg, refNode);\n"
        "        }\n"
        "        if (wrapper.children.length === 0) {\n"
        "          wrapper.remove();\n"
        "        }\n"
        "      } else {\n"
        "        // If split-agent-context exists, wrap it\n"
        "        if (splitContext && !splitContext.classList.contains('merged')) {\n"
        "          const splitWrapper = document.createElement('div');\n"
        "          splitWrapper.className = 'simultaneous-messages';\n"
        "          const splitUnmergeBtn = document.createElement('button');\n"
        "          splitUnmergeBtn.className = 'unmerge-btn';\n"
        "          splitUnmergeBtn.innerHTML = '✕';\n"
        "          splitUnmergeBtn.title = 'Click to unmerge messages';\n"
        "          splitWrapper.appendChild(splitUnmergeBtn);\n"
        "          splitWrapper.dataset.isSplitContext = 'true';\n"
        "          parent.insertBefore(splitWrapper, splitContext);\n"
        "          splitWrapper.appendChild(splitContext);\n"
        "          splitContext.classList.add('merged');\n"
        "        }\n"
        "        \n"
        "        // Create wrapper for prompts if both exist\n"
        "        if (currentPrompt && nextPrompt) {\n"
        "          const promptWrapper = document.createElement('div');\n"
        "          promptWrapper.className = 'simultaneous-messages';\n"
        "          const promptUnmergeBtn = document.createElement('button');\n"
        "          promptUnmergeBtn.className = 'unmerge-btn';\n"
        "          promptUnmergeBtn.innerHTML = '✕';\n"
        "          promptUnmergeBtn.title = 'Click to unmerge messages';\n"
        "          promptWrapper.appendChild(promptUnmergeBtn);\n"
        "          promptWrapper.dataset.firstMsgId = currentPrompt.dataset.msgId;\n"
        "          promptWrapper.dataset.secondMsgId = nextPrompt.dataset.msgId;\n"
        "          \n"
        "          // Determine order: agent-0 first, agent-1 second\n"
        "          const firstPrompt = currentPrompt.classList.contains('agent-0') ? currentPrompt : nextPrompt;\n"
        "          const secondPrompt = currentPrompt.classList.contains('agent-0') ? nextPrompt : currentPrompt;\n"
        "          \n"
        "          parent.insertBefore(promptWrapper, currentPrompt);\n"
        "          promptWrapper.appendChild(firstPrompt);\n"
        "          promptWrapper.appendChild(secondPrompt);\n"
        "          currentPrompt.classList.add('merged');\n"
        "          nextPrompt.classList.add('merged');\n"
        "        }\n"
        "        \n"
        "        // Create wrapper for assistant messages\n"
        "        const wrapper = document.createElement('div');\n"
        "        wrapper.className = 'simultaneous-messages';\n"
        "        const unmergeBtn = document.createElement('button');\n"
        "        unmergeBtn.className = 'unmerge-btn';\n"
        "        unmergeBtn.innerHTML = '✕';\n"
        "        unmergeBtn.title = 'Click to unmerge messages';\n"
        "        wrapper.appendChild(unmergeBtn);\n"
        "        wrapper.dataset.firstMsgId = currentMsg.dataset.msgId;\n"
        "        wrapper.dataset.secondMsgId = nextMsg.dataset.msgId;\n"
        "        \n"
        "        // Determine order: agent-0 first, agent-1 second\n"
        "        const firstAssistant = currentMsg.classList.contains('agent-0') ? currentMsg : nextMsg;\n"
        "        const secondAssistant = currentMsg.classList.contains('agent-0') ? nextMsg : currentMsg;\n"
        "        \n"
        "        parent.insertBefore(wrapper, currentMsg);\n"
        "        wrapper.appendChild(firstAssistant);\n"
        "        wrapper.appendChild(secondAssistant);\n"
        "        currentMsg.classList.add('merged');\n"
        "        nextMsg.classList.add('merged');\n"
        "      }\n"
        "    }\n"
        "    if (e.target.classList.contains('unmerge-btn')) {\n"
        "      const wrapper = e.target.closest('.simultaneous-messages');\n"
        "      if (!wrapper) return;\n"
        "      const parent = wrapper.parentElement;\n"
        "      \n"
        "      // Check if this is a split-context wrapper\n"
        "      if (wrapper.dataset.isSplitContext === 'true') {\n"
        "        const splitContext = wrapper.querySelector('.split-agent-context');\n"
        "        if (splitContext) {\n"
        "          splitContext.classList.remove('merged');\n"
        "          parent.insertBefore(splitContext, wrapper.nextElementSibling);\n"
        "        }\n"
        "        wrapper.remove();\n"
        "        return;\n"
        "      }\n"
        "      \n"
        "      const firstMsgId = wrapper.dataset.firstMsgId;\n"
        "      const secondMsgId = wrapper.dataset.secondMsgId;\n"
        "      const messages = Array.from(wrapper.querySelectorAll('.chat-message'));\n"
        "      const refNode = wrapper.nextElementSibling;\n"
        "      const firstMsg = messages.find(m => m.dataset.msgId === firstMsgId);\n"
        "      const secondMsg = messages.find(m => m.dataset.msgId === secondMsgId);\n"
        "      \n"
        "      // Check for preceding wrappers to also unmerge (prompts and split-context)\n"
        "      let currentElem = wrapper.previousElementSibling;\n"
        "      const wrappersToUnmerge = [];\n"
        "      \n"
        "      while (currentElem) {\n"
        "        if (currentElem.classList.contains('simultaneous-messages')) {\n"
        "          wrappersToUnmerge.push(currentElem);\n"
        "        } else if (currentElem.classList.contains('chat-message') || currentElem.classList.contains('chat-group-divider')) {\n"
        "          break;\n"
        "        }\n"
        "        currentElem = currentElem.previousElementSibling;\n"
        "      }\n"
        "      \n"
        "      // Unmerge preceding wrappers\n"
        "      for (const prevWrapper of wrappersToUnmerge) {\n"
        "        if (prevWrapper.dataset.isSplitContext === 'true') {\n"
        "          const splitContext = prevWrapper.querySelector('.split-agent-context');\n"
        "          if (splitContext) {\n"
        "            splitContext.classList.remove('merged');\n"
        "            parent.insertBefore(splitContext, prevWrapper.nextElementSibling);\n"
        "          }\n"
        "          prevWrapper.remove();\n"
        "        } else {\n"
        "          const prevMessages = Array.from(prevWrapper.querySelectorAll('.chat-message'));\n"
        "          const prevFirstMsgId = prevWrapper.dataset.firstMsgId;\n"
        "          const prevSecondMsgId = prevWrapper.dataset.secondMsgId;\n"
        "          const prevFirstMsg = prevMessages.find(m => m.dataset.msgId === prevFirstMsgId);\n"
        "          const prevSecondMsg = prevMessages.find(m => m.dataset.msgId === prevSecondMsgId);\n"
        "          const prevRefNode = prevWrapper.nextElementSibling;\n"
        "          \n"
        "          if (prevFirstMsg) {\n"
        "            prevFirstMsg.classList.remove('merged');\n"
        "            prevFirstMsg.style.display = '';\n"
        "            parent.insertBefore(prevFirstMsg, prevRefNode);\n"
        "          }\n"
        "          if (prevSecondMsg) {\n"
        "            prevSecondMsg.classList.remove('merged');\n"
        "            prevSecondMsg.style.display = '';\n"
        "            parent.insertBefore(prevSecondMsg, prevRefNode);\n"
        "          }\n"
        "          prevWrapper.remove();\n"
        "        }\n"
        "      }\n"
        "      \n"
        "      // Unmerge the main assistant messages\n"
        "      if (firstMsg) {\n"
        "        firstMsg.classList.remove('merged');\n"
        "        firstMsg.style.display = '';\n"
        "        parent.insertBefore(firstMsg, refNode);\n"
        "      }\n"
        "      if (secondMsg) {\n"
        "        secondMsg.classList.remove('merged');\n"
        "        secondMsg.style.display = '';\n"
        "        parent.insertBefore(secondMsg, refNode);\n"
        "      }\n"
        "      wrapper.remove();\n"
        "    }\n"
        "  });\n"
        "});\n"
        "</script>",
        "</head>",
        "<body>",
        '<div class="toolbar-wrap">',
        '<div class="toolbar-hotzone"></div>',
        '<div class="toolbar">',
        '<button id="toggle-strong-hide"><span class="emoji-bw">🗜️</span> Strong Hide: <span id="strong-hide-state">Off</span></button>',
        '<button id="toggle-hide-user-messages"><span class="emoji-bw">👁️</span> Hide Prompts: <span id="hide-user-state">Off</span></button>',
        '<span id="chat-width-control" style="margin-left:8px;">',
        '<label for="chat-width-slider"><span class="emoji-bw">↔️</span> Width:</label>',
        '<input id="chat-width-slider" type="range" min="600" max="1600" step="50" value="900" style="width:120px; vertical-align:middle;" />',
        '<span id="chat-width-value" style="margin-left:4px;">900px</span>',
        "</span>",
        '<span style="margin-left:12px;">',
        '<label for="font-family-select"><span class="emoji-bw">🔤</span> Font:</label>',
        '<select id="font-family-select" style="padding:2px 6px; border:1px solid var(--accent-muted); border-radius:var(--corner-radius); background:var(--bg);">',
        "<option value=\"'Segoe UI', Tahoma, Geneva, Verdana, sans-serif\">Segoe UI</option>",
        '<option value="Arial, sans-serif">Arial</option>',
        "<option value=\"'Helvetica Neue', Helvetica, sans-serif\">Helvetica</option>",
        "<option value=\"'Times New Roman', Times, serif\">Times New Roman</option>",
        '<option value="Georgia, serif">Georgia</option>',
        "<option value=\"'Courier New', Courier, monospace\">Courier New</option>",
        "<option value=\"'Comic Sans MS', cursive\">Comic Sans</option>",
        "<option value=\"'Trebuchet MS', sans-serif\">Trebuchet MS</option>",
        '<option value="Verdana, sans-serif">Verdana</option>',
        "<option value=\"'Palatino Linotype', 'Book Antiqua', Palatino, serif\">Palatino</option>",
        "<option value=\"'Lucida Console', Monaco, monospace\">Lucida Console</option>",
        "</select>",
        "</span>",
        '<span style="margin-left:8px;">',
        '<label for="font-size-input"><span class="emoji-bw">📏</span> Size:</label>',
        '<input id="font-size-input" type="number" min="8" max="24" step="1" value="14" style="width:50px;" />',
        "<span>px</span>",
        "</span>",
        '<span style="margin-left:12px; display:flex; align-items:center; gap:8px;">',
        '<label style="font-weight:600;">Agent Names:</label>',
        f'<select id="agent0-emoji-input" style="width:65px; padding:2px 6px; border:1px solid var(--accent-muted); border-radius:var(--corner-radius); background:var(--bg);">',
        '<option value="🤖">🤖 Robot</option>',
        '<option value="👤">👤 Human</option>',
        "</select>",
        f'<input id="agent0-name-input" type="text" placeholder="{html.escape(unique_agent_ids[0]) if len(unique_agent_ids) > 0 else "Agent 0"}" style="width:80px; padding:2px 6px; border:1px solid var(--accent-muted); border-radius:var(--corner-radius); background:var(--bg);" />',
        '<span style="margin:0 4px;">|</span>',
        f'<select id="agent1-emoji-input" style="width:65px; padding:2px 6px; border:1px solid var(--accent-muted); border-radius:var(--corner-radius); background:var(--bg);">',
        '<option value="🤖">🤖 Robot</option>',
        '<option value="👤">👤 Human</option>',
        "</select>",
        f'<input id="agent1-name-input" type="text" placeholder="{html.escape(unique_agent_ids[1]) if len(unique_agent_ids) > 1 else "Agent 1"}" style="width:80px; padding:2px 6px; border:1px solid var(--accent-muted); border-radius:var(--corner-radius); background:var(--bg);" />',
        '<button id="apply-agent-names" style="padding:4px 8px; border:1px solid var(--accent-muted); background:var(--panel-bg); border-radius:var(--corner-radius); cursor:pointer;">Apply</button>',
        "</span>",
        "</div>",
        "</div>",
    ]

    # Add Chat View
    import html as _html_mod

    html_parts.append('<div id="flow-chat" class="messages-flow">')

    # Helper function to add context annotation areas
    def add_context_area(position: str, time_step: int):
        context_key = f"round-context-{position}-{time_step}"
        placeholder = f"Add context {position} round {time_step}..."
        color_buttons = ""
        # Add default/reset color button first
        color_buttons += (
            f'<div class="context-color-btn" data-color="default" '
            f'style="background: linear-gradient(135deg, #000 25%, transparent 25%, transparent 75%, #000 75%), '
            f"linear-gradient(135deg, #000 25%, transparent 25%, transparent 75%, #000 75%); "
            f"background-size: 4px 4px; background-position: 0 0, 2px 2px; "
            f'background-color: #fff;" title="Default color"></div>'
        )
        for color_name, color_value in [
            ("red", "#d32f2f"),
            ("orange", "#f57c00"),
            ("yellow", "#f9a825"),
            ("green", "#388e3c"),
            ("blue", "#1976d2"),
            ("purple", "#7b1fa2"),
            ("gray", "#666666"),
        ]:
            color_buttons += (
                f'<div class="context-color-btn" data-color="{color_value}" '
                f'style="background-color: {color_value};" title="{color_name}"></div>'
            )

        html_parts.append(
            f'<div class="round-context">'
            f'<div class="round-context-edit" contenteditable="true" spellcheck="true" '
            f'data-context-key="{context_key}" '
            f'data-placeholder="{placeholder}"></div>'
            f'<div class="round-context-controls">{color_buttons}</div>'
            f"</div>"
        )

    # Helper function to add split agent context boxes
    def add_split_agent_contexts(position: str, time_step: int):
        color_buttons = ""
        # Add default/reset color button first
        color_buttons += (
            f'<div class="context-color-btn" data-color="default" '
            f'style="background: linear-gradient(135deg, #000 25%, transparent 25%, transparent 75%, #000 75%), '
            f"linear-gradient(135deg, #000 25%, transparent 25%, transparent 75%, #000 75%); "
            f"background-size: 4px 4px; background-position: 0 0, 2px 2px; "
            f'background-color: #fff;" title="Default color"></div>'
        )
        for color_name, color_value in [
            ("red", "#d32f2f"),
            ("orange", "#f57c00"),
            ("yellow", "#f9a825"),
            ("green", "#388e3c"),
            ("blue", "#1976d2"),
            ("purple", "#7b1fa2"),
            ("gray", "#666666"),
        ]:
            color_buttons += (
                f'<div class="context-color-btn" data-color="{color_value}" '
                f'style="background-color: {color_value};" title="{color_name}"></div>'
            )

        html_parts.append('<div class="split-agent-context">')

        # Agent 0 box
        agent0_key = f"agent-context-0-{position}-{time_step}"
        agent0_placeholder = f"..."
        html_parts.append(
            f'<div class="agent-context-box agent-0">'
            f'<div class="round-context-edit" contenteditable="true" spellcheck="true" '
            f'data-context-key="{agent0_key}" '
            f'data-placeholder="{agent0_placeholder}"></div>'
            f'<div class="round-context-controls">{color_buttons}</div>'
            f"</div>"
        )

        # Agent 1 box
        agent1_key = f"agent-context-1-{position}-{time_step}"
        agent1_placeholder = f"..."
        html_parts.append(
            f'<div class="agent-context-box agent-1">'
            f'<div class="round-context-edit" contenteditable="true" spellcheck="true" '
            f'data-context-key="{agent1_key}" '
            f'data-placeholder="{agent1_placeholder}"></div>'
            f'<div class="round-context-controls">{color_buttons}</div>'
            f"</div>"
        )

        html_parts.append("</div>")  # split-agent-context

    last_time_step_chat = None
    for original_index, turn in indexed_turns:
        # Use agent index for CSS class (agent-0 or agent-1) instead of agent ID
        agent_index = agent_id_to_index.get(turn.agent_id, 0)
        agent_class = f"agent-{agent_index}"
        role_class = f"role-{turn.role}"

        # Add time step divider and beginning context
        if last_time_step_chat is None or turn.time_step != last_time_step_chat:
            # Add end contexts for previous round (only regular context, not prompt summary)
            if last_time_step_chat is not None:
                add_context_area("end", last_time_step_chat)

            html_parts.append(
                f'<div class="chat-group-divider">'
                f'<span class="chat-group-label">⏱ Round {turn.time_step + 1}</span>'
                f"</div>"
            )

            # Add beginning contexts for new round (both context and prompt summary)
            add_context_area("beginning", turn.time_step)
            add_split_agent_contexts("beginning", turn.time_step)

            last_time_step_chat = turn.time_step

        # Build chat message with merge controls
        html_parts.append(
            f'<div class="chat-message {agent_class} {role_class}" data-msg-id="{original_index}">'
        )

        # Add merge control button
        html_parts.append(
            f'<button class="merge-btn" title="Merge with next message" data-msg-id="{original_index}">⇄</button>'
        )

        html_parts.append('<div class="chat-message-content">')

        # Header with agent name and reward (always show reward)
        if turn.role == "assistant":
            name = _html_mod.escape(turn.agent_id)
            raw_val = turn.reward
            if isinstance(raw_val, (int, float)):
                reward_val = f"{raw_val:.4f}".rstrip("0").rstrip(".")
                if len(reward_val) > 8:
                    reward_val = reward_val[:8] + "…"
            else:
                reward_val = str(raw_val)
            header_html = (
                f'<div class="chat-header">'
                f'<span class="emoji-bw" data-agent-index="{agent_index}">🤖</span> <span class="agent-name" data-agent-index="{agent_index}">{name}</span>'
                f'<span class="chat-reward">⚑ {reward_val}</span>'
                f"</div>"
            )
        else:
            name = _html_mod.escape(turn.agent_id)
            header_html = f'<div class="chat-header">Prompt of <span class="agent-name" data-agent-index="{agent_index}">{name}</span></div>'

        html_parts.append(header_html)

        # Reasoning content if present
        if turn.reasoning_content:
            _raw_reasoning = turn.reasoning_content.replace("\r\n", "\n")
            _raw_reasoning = _re.sub(r"^\s*\n+", "", _raw_reasoning)
            esc_reasoning = _html_mod.escape(_raw_reasoning)
            html_parts.append(
                f'<div class="chat-reasoning collapsed">'
                f'<span class="reasoning-icon">💭</span> '
                f'<span class="reasoning-text">{esc_reasoning}</span>'
                f"</div>"
            )

        # Message bubble
        esc_content = _html_mod.escape(turn.content)
        html_parts.append(f'<div class="chat-bubble">{esc_content}</div>')

        html_parts.append("</div>")  # chat-message-content
        html_parts.append("</div>")  # chat-message

    # Add end contexts for the last round (only regular context, not prompt summary)
    if last_time_step_chat is not None:
        add_context_area("end", last_time_step_chat)

    html_parts.append("</div>")  # flow-chat
    html_parts.extend(["</body>", "</html>"])

    return "\n".join(html_parts)


def export_html_from_rollout_tree(path: Path, outdir: Path, main_only: bool = False):
    """Process a rollout tree file and generate HTML files for each path.
    Creates separate HTML files for the main path and each branch path.
    The main path is saved in the root output directory, while branch paths
    are saved in a 'branches' subdirectory.

    Args:
        path: Path to the rollout tree JSON file
        outdir: Output directory for HTML files
        main_only: If True, only export the main trajectory (default: False)
    """
    root = load_rollout_tree(path)
    mgid = root.id

    main_path, branch_paths = get_rollout_tree_paths(root)

    outdir.mkdir(parents=True, exist_ok=True)

    # Create branches subdirectory if we have branch paths
    if not main_only and branch_paths:
        branches_dir = outdir / f"mgid:{mgid}_branches_html_renders"
        branches_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML for the main path
    chat_turns = gather_all_chat_turns_for_path(main_path)
    html_content = html_from_chat_turns(chat_turns)
    output_file = outdir / f"mgid:{mgid}_main_html_render.render.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate HTML for each branch path
    for path_obj in branch_paths:
        chat_turns = gather_all_chat_turns_for_path(path_obj)

        html_content = html_from_chat_turns(chat_turns)

        path_id: str = path_obj.id
        output_filename = f"{path_id}_html_render.render.html"

        output_file = branches_dir / output_filename

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
