import subprocess
from logging_function import setup_logger
import nltk
import logging

#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')

if __name__ == '__main__':
    setup_logger(app_name="AI_DM_RAG")  # This sets up logging, but returns nothing
    logger = logging.getLogger(__name__)


    logger.info("Starting the Dungeon Master API and Flask app...")
    # Start the Flask app and Dungeon Master API
    subprocess.Popen(['python', 'chatbox_server.py'])
    subprocess.Popen(['python', 'dungeon_master_api.py'])
