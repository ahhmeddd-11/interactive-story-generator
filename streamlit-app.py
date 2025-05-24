import streamlit as st
from huggingface_hub import InferenceClient
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
STORY_THEMES = [
    "Adventure", "Mystery", "Romance", "Historical", "Slice of Life", "Fairy Tale"
]
CHARACTER_TEMPLATES = {
    "Adventurer": "A brave and fearless explorer who loves adventure and challenges.",
    "Detective": "A keen and observant detective skilled in observation and deduction.",
    "Artist": "A creative artist with unique perspectives on beauty.",
    "Scientist": "A curious scientist dedicated to exploring the unknown.",
    "Ordinary Person": "An ordinary person with a rich inner world."
}
STORY_STYLES = [
    "Fantasy", "Science Fiction", "Mystery", "Adventure", "Romance", "Horror"
]
STORY_SYSTEM_PROMPT = """You are a professional story generator. Your task is to generate coherent and engaging stories based on user settings and real-time input.

Key requirements:
1. The story must maintain continuity, with each response building upon all previous plot developments
2. Carefully analyze dialogue history to maintain consistency in character personalities and plot progression
3. Naturally integrate new details or development directions when provided by the user
4. Pay attention to cause and effect, ensuring each plot point has reasonable setup and explanation
5. Make the story more vivid through environmental descriptions and character dialogues
6. At key story points, provide hints to guide user participation in plot progression

You should not:
1. Start a new story
2. Ignore previously mentioned important plots or details
3. Generate content that contradicts established settings
4. Introduce major turns without proper setup

Remember: You are creating an ongoing story, not independent fragments."""

# Hugging Face Client
def create_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set")
    return InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=token)

# Story Generation
def generate_story(prompt, history, temperature, max_tokens, top_p):
    client = create_client()
    messages = [{"role": "system", "content": STORY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        full_response = ""
        for message in client.chat_completion(messages, max_tokens=max_tokens, stream=True,
                                              temperature=temperature, top_p=top_p):
            if hasattr(message.choices[0].delta, 'content'):
                token = message.choices[0].delta.content
                if token:
                    full_response += token
                    yield full_response
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Interactive Story Generator", layout="wide")
st.title("🎭 Interactive Story Generator")
st.markdown("Prepared by: Syed Ahmed Ali")
st.markdown("Let AI create a unique storytelling experience for you. Choose your story style, theme, add character settings, then describe a scene to start your story. Interact with AI to continue developing the plot!")

# Tabs
tab1, tab2 = st.tabs(["✍️ Story Creation", "⚙️ Advanced Settings"])

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Advanced settings
with tab2:
    temperature = st.slider("Creativity (Temperature)", 0.1, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Maximum Generation Length", 64, 1024, 512, 64)
    top_p = st.slider("Sampling Range (Top-p)", 0.1, 1.0, 0.95, 0.05)

# Story creation tab
with tab1:
    with st.sidebar:
        style = st.selectbox("Choose Story Style", STORY_STYLES)
        theme = st.selectbox("Choose Story Theme", STORY_THEMES)
        character_template = st.selectbox("Character Template", list(CHARACTER_TEMPLATES.keys()))
        if "last_template" not in st.session_state or st.session_state.last_template != character_template:
            st.session_state.character_desc = CHARACTER_TEMPLATES[character_template]
            st.session_state.last_template = character_template
        character_desc = st.text_area("Character Description", value=st.session_state.get("character_desc", ""))
        scene = st.text_area("Scene Description", height=150)

        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.experimental_rerun()

    if st.button("✨ Start / Continue Story"):
        past_story = "\n".join([h["assistant"] for h in st.session_state.history if "assistant" in h])
        if past_story:
            prompt = f"Previously in the story:\n---\n{past_story}\n---\n\nUser's new input: {scene}"
        else:
            prompt = f"Style: {style}\nTheme: {theme}\nCharacter: {character_desc}\nScene: {scene}"

        st.session_state.history.append({"user": scene})
        response_container = st.empty()
        result = ""
        for chunk in generate_story(prompt, st.session_state.history, temperature, max_tokens, top_p):
            result = chunk
            response_container.markdown(result)
        st.session_state.history[-1]["assistant"] = result

    st.markdown("---")
    st.subheader("📚 Story Dialogue")
    for i, chat in enumerate(st.session_state.history):
        with st.expander(f"Turn {i+1}"):
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**AI:** {chat['assistant']}")

    if st.button("💾 Save Story"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("stories", exist_ok=True)
        filename = f"stories/story_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== Interactive Story ===\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Style: {style}\nTheme: {theme}\nCharacter: {character_desc}\n\n")
            for i, chat in enumerate(st.session_state.history):
                f.write(f"--- Turn {i+1} ---\nUser: {chat['user']}\nAI: {chat['assistant']}\n\n")
        st.success(f"✅ Story saved as: {filename}")

with st.expander("📖 Usage Guide"):
    st.markdown("""
    ### How to Use
    1. Select a story style and theme.
    2. Choose a character template or write your own character description.
    3. Provide a scene description and click **Start / Continue Story**.
    4. Interact to advance the story or clear to start fresh.

    ### Parameters
    - **Temperature:** Controls creativity (higher = more creative).
    - **Top-p:** Controls vocabulary richness (higher = more diverse).
    - **Max Tokens:** Maximum length of each AI reply.

    ### Tips
    - Use detailed descriptions to make the story vivid.
    - Save your best stories to revisit or share.
    """)
