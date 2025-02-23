#!/usr/bin/env python3
import threading
import asyncio
import time
from flask import Flask, request, jsonify
import freeresearch_core  # Import the module to reference its globals
from freeresearch_core import run_full_research, logs_data, fetched_links_data, total_tokens_fetched

app = Flask(__name__)

# Global variables for tracking research progress and final report
research_in_progress = False
final_report_result = None

def add_log_entry(sender, content):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sender": sender,
        "content": content
    }
    logs_data.append(entry)

def background_research(subject):
    global research_in_progress, final_report_result
    research_in_progress = True
    # Create a new asyncio event loop for this background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    add_log_entry("System", f"Research started for subject: {subject}")
    try:
        final_report = run_full_research(subject)
        final_report_result = final_report
        add_log_entry("System", "Research completed successfully.")
    except Exception as e:
        final_report_result = f"Error: {str(e)}"
        add_log_entry("System", f"Error occurred: {str(e)}")
    finally:
        research_in_progress = False
        loop.close()

@app.route("/api/research", methods=["POST"])
def start_research():
    global research_in_progress, final_report_result
    data = request.get_json(force=True)
    subject = data.get("subject")
    if not subject:
        return jsonify({"error": "No subject provided"}), 400
    if research_in_progress:
        return jsonify({"error": "Research is already in progress"}), 429

    # Reset globals for a fresh research session
    from freeresearch_core import reset_globals
    reset_globals()

    final_report_result = None
    research_in_progress = True
    logs_data.clear()
    threading.Thread(target=background_research, args=(subject,), daemon=True).start()
    return jsonify({"message": "Research started successfully"}), 202

@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(list(logs_data))

@app.route("/api/report", methods=["GET"])
def get_report():
    if final_report_result:
        return jsonify({"report": final_report_result})
    else:
        return jsonify({"report": None})

@app.route("/api/links", methods=["GET"])
def get_links():
    return jsonify(fetched_links_data)

@app.route("/api/knowledge-graph/session", methods=["GET"])
def get_session_knowledge_graph():
    return jsonify(freeresearch_core.knowledge_graph_data)

@app.route("/api/knowledge-graph/persistent", methods=["GET"])
def get_persistent_knowledge_graph():
    return jsonify(freeresearch_core.persistent_knowledge_graph_data)

@app.route("/api/knowledge-graph", methods=["GET"])
def get_knowledge_graph():
    return jsonify({
        "session": freeresearch_core.knowledge_graph_data,
        "persistent": freeresearch_core.persistent_knowledge_graph_data
    })

@app.route("/api/agent-conversation", methods=["GET"])
def get_agent_conversation():
    return jsonify(list(logs_data))

@app.route("/api/resources", methods=["GET"])
def get_resources():
    resources = {
        "cpu_usage": 0,
        "memory_usage": 0,
        "api_calls": 0,
        "active_tasks": 0,
        "total_tokens_fetched": total_tokens_fetched
    }
    return jsonify(resources)

if __name__ == "__main__":
    # For production, consider a WSGI server like gunicorn or uwsgi.
    # For development, debug=True is fine.
    app.run(host="0.0.0.0", port=5000, debug=True)
