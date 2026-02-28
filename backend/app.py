from flask import Flask,request
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import tempfile
import assemblyai as aai
import os
import base64
import requests
import json
from flask import jsonify

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY

checkpointer = InMemorySaver()

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer
)

question_count = 0
current_subject = ""
thread_id = "interview_session"

INTERVIEW_PROMPT = """You are Natalie, a friendly and conversational interviewer conducting a natural {subject} interview.

IMPORTANT GUIDELINES:
1. Ask exactly 5 questions total throughout the interview
2. Keep questions SHORT and CRISP (1-2 sentences maximum)
3. ALWAYS reference what the candidate ACTUALLY said in their previous answer - do NOT make up or assume their answers
4. Show genuine interest with brief acknowledgments based on their REAL responses
5. Adapt questions based on their ACTUAL responses - go deeper if they're strong, adjust if uncertain
6. Be warm and conversational but CONCISE
7. No lengthy explanations - just ask clear, direct questions

CRITICAL: Read the conversation history carefully. Only acknowledge what the candidate truly said, not what you think they might have said.

Keep it short, conversational, and adaptive!"""

FEEDBACK_PROMPT = """Based on our complete interview conversation, provide detailed feedback as JSON only:
    {{
    "subject": "<topic>",
    "candidate_score": <1-5>,
    "feedback": "<detailed strengths with specific examples 
    from their ACTUAL answers>",
    "areas_of_improvement": "<constructive suggestions based 
    on gaps you noticed>"
    }}
    Be specific - reference ACTUAL things they said during the interview."""



app = Flask(__name__)
CORS(app,expose_headers=['X-Question-Number'])


def stream_audio(text):
    BASE_URL = "https://global.api.murf.ai/v1/speech/stream"
    payload = {
        "text": text,
        "voiceId": "en-US-natalie",
        "model": "FALCON",
        "multiNativeLocale": "en-US",
        "sampleRate": 24000,
        "format": "MP3",
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }
    response = requests.post(
        BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        stream=True
    )
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            yield base64.b64encode(chunk).decode("utf-8") + "\n"


def speech_to_text(audio_path):
    try:
        config = aai.TranscriptionConfig(
            speech_models=["universal"]  # MUST be a list
        )

        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            print("Transcription error:", transcript.error)
            return ""

        return transcript.text or ""

    except Exception as e:
        print("Speech-to-text failed:", str(e))
        return ""




@app.route("/start-interview", methods=["POST"])
def start_interview():
    global question_count, current_subject, checkpointer, agent
    data = request.json
    current_subject = data.get("subject", "Python")
    question_count = 1
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer
    )
    config = {"configurable": {"thread_id": thread_id}}
    formatted_prompt = INTERVIEW_PROMPT.format(subject=current_subject)
    response = agent.invoke({
        "messages": [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"Start the interview with a warm greeting and ask the first question about {current_subject}. Keep it SHORT (1-2 sentences)."}
        ]
    }, config=config)
    question = response["messages"][-1].content
    print(f"\n[Question {question_count}] {question}")
    return stream_audio(question), {"Content-Type": "text/plain"} 


@app.route("/submit-answer", methods=["POST"])
def submit_answer():    
    global question_count 
    audio_file = request.files["audio"] 
    question_count += 1
    temp_path = (tempfile.NamedTemporaryFile(delete=False, suffix=".webm")).name
    audio_file.save(temp_path)
    answer = speech_to_text(temp_path) 
    
    os.unlink(temp_path) 
    if not answer:
        answer = "Empty answer received." 
    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke({"messages": [{"role": "system", "content": answer}]}, config=config)
    prompt = f"""The candidate just answered question {question_count - 1}.
 
    Look at their ACTUAL answer above. Do NOT assume or make up what they said.
    
    Now ask question {question_count} of 5:
    1. Briefly acknowledge what they ACTUALLY said (1 sentence) - quote their exact words if needed
    2. Ask your next question that builds on their REAL response (1-2 sentences)
    3. If they said "I don't know" or gave a wrong answer, acknowledge that and ask something simpler
    4. Keep the TOTAL response under 3 sentences
    
    Be conversational but CONCISE. Only reference what they truly said."""
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    question = response["messages"][-1].content
    print(f"\n[Question {question_count}] {question}") 
    return (stream_audio(question),
        {
        'Content-Type': 'text/plain',
        'X-Question-Number': str(question_count)
        })
    
@app.route("/get-feedback", methods=["POST"]) 
def get_feedback():
    """Generate detailed interview feedback"""
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke({
        "messages": [
        {
            "role": "user", 
            "content": f"{FEEDBACK_PROMPT}\n\nReview our complete {current_subject} interview conversation and provide detailed feedback."
        }
        ]
    }, config=config)
    text = response["messages"][-1].content
    print(f"\n[Feedback Generated]\n{text}\n")

    feedback = None
    try:
        cleaned = text.strip()

        # Strategy 1: extract from ```json ... ``` block
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        # Strategy 2: extract from ``` ... ``` block
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].replace("json", "", 1).strip()
        # Strategy 3: find first { and last } to extract raw JSON
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end != 0:
                cleaned = cleaned[start:end]

        feedback = json.loads(cleaned)
    except Exception as e:
        print(f"[Feedback Parse Error] {e}\nRaw text:\n{text}")
        return jsonify({"success": False, "error": f"Failed to parse feedback: {str(e)}", "raw": text}), 500

    return jsonify({"success": True, "feedback": feedback})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)