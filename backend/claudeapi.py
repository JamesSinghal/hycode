import os
from typing import Dict, List, Optional
from datetime import datetime
from anthropic import Anthropic
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document
import json

# Constants
ANTHROPIC_API_KEY = os.getenv('API_KEY')
MODEL_NAME = "claude-3-haiku-20240307"

class ClaudeChatbot:
    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = MODEL_NAME):
        """
        Initialize the Claude chatbot with conversation history storage.
        
        Args:
            api_key (str): Anthropic API key
            model (str): Claude model to use
        """
        # Initialize Anthropic client with explicit API key
        self.client = Anthropic(
            api_key=api_key,
        )
        self.model = model
        self.max_response_chars = 2000
        
        # Initialize Haystack document store for conversation history
        self.document_store = InMemoryDocumentStore(
            index="conversations",
            embedding_dim=768
        )
        
        # Keep track of current conversation
        self.current_conversation_id = None
        self.messages = []

    def start_new_conversation(self) -> str:
        """Start a new conversation and return the conversation ID."""
        self.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages = []
        return self.current_conversation_id

    def _store_message(self, role: str, content: str):
        """Store a message in the current conversation history."""
        timestamp = datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.messages.append(message)
        
        # Store in Haystack
        doc = Document(
            content=json.dumps(message),
            meta={
                "conversation_id": self.current_conversation_id,
                "timestamp": timestamp,
                "role": role
            }
        )
        self.document_store.write_documents([doc])

    def chat(self, user_input: str) -> str:
        """
        Send a message to Claude and get a response.
        
        Args:
            user_input (str): User's message
            
        Returns:
            str: Claude's response
        """
        if not self.current_conversation_id:
            self.start_new_conversation()

        # Store user message
        self._store_message("user", user_input)

        try:
            # Create message for Claude using the latest API format
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1024
            )

            assistant_message = response.content[0].text
            
            # Store assistant message
            self._store_message("assistant", assistant_message)
            
            return assistant_message
            
        except Exception as e:
            print(f"Error communicating with Claude: {str(e)}")
            return f"Error: {str(e)}"

    def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve conversation history for a specific conversation ID.
        
        Args:
            conversation_id (str, optional): Conversation ID to retrieve. 
                                          If None, returns current conversation.
                                          
        Returns:
            List[Dict]: List of messages in the conversation
        """
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            return []

        # Query Haystack for conversation messages
        docs = self.document_store.get_all_documents(
            filters={"conversation_id": conv_id},
            return_embedding=False
        )
        
        # Sort messages by timestamp
        messages = [json.loads(doc.content) for doc in docs]
        messages.sort(key=lambda x: x["timestamp"])
        
        return messages

    def search_conversations(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search through conversation history using semantic search.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant messages
        """
        # Convert all documents to format suitable for searching
        docs = self.document_store.get_all_documents()
        results = []
        
        for doc in docs:
            message = json.loads(doc.content)
            if query.lower() in message["content"].lower():
                results.append({
                    "conversation_id": doc.meta["conversation_id"],
                    "message": message,
                    "score": 1.0 if query.lower() in message["content"].lower() else 0.0
                })
        
        # Sort by score and return top_k results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Example usage
if __name__ == "__main__":
    # Initialize chatbot with the constant API key
    chatbot = ClaudeChatbot()
    
    # Start new conversation
    conv_id = chatbot.start_new_conversation()
    
    # Example interaction
    try:
        response = chatbot.chat("Hello! Can you help me understand quantum computing?")
        print(f"Claude: {response}")
        
        # Get conversation history
        history = chatbot.get_conversation_history()
        print("\nConversation History:")
        for message in history:
            print(f"{message['role']}: {message['content']}")
        
        # Search conversations
        search_results = chatbot.search_conversations("quantum")
        print("\nSearch Results for 'quantum':")
        for result in search_results:
            print(f"Conversation {result['conversation_id']}: {result['message']['content']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")