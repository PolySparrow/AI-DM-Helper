import subprocess
import logging_function
import nltk

#nltk.download('wordnet')
#nltk.download('omw-1.4')

if __name__ == '__main__':
    logger= logging_function.setup_logger()


    logger.info("Starting the Dungeon Master API and Flask app...")
    # Start the Flask app and Dungeon Master API
    subprocess.Popen(['python', 'chatbox_server.py'])
    subprocess.Popen(['python', 'dungeon_master_api.py'])
