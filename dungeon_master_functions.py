import openai
import logging
import config

def CreateOpenAIClient(openai_apikey=config.openai_apikey):
    """Create and return an OpenAI client."""
    return openai.OpenAI(openai_apikey)

def Chat(prompt):
    AI=CreateOpenAIClient()
    """Send a prompt to the OpenAI API and return the response."""
    try:
        response = AI.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        message=response.choices[0].message.content.strip()
        AI.close()
        return message
    except Exception as e:
        logging.error(f"Error in Chat function: {e}")
        return str(e)
    

