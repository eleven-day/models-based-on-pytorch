import re

SYSTEM_PROMPT = "You are an AI assistant that can understand images and have a conversation about them. Answer the questions in a helpful manner."

class Conversation:
    def __init__(self, system=SYSTEM_PROMPT):
        self.system = system
        self.messages = []
        self.image_placeholder = "<image>"
    
    def add_message(self, role, content, image=None):
        message = {"role": role, "content": content}
        if image is not None and role == "user":
            message["image"] = image
            # Replace placeholder with actual image reference
            message["content"] = message["content"].replace(self.image_placeholder, "<image_token>")
        self.messages.append(message)
    
    def get_prompt(self):
        prompt = f"[SYSTEM] {self.system}\n"
        for message in self.messages:
            if message["role"] == "user":
                prompt += f"[USER] {message['content']}\n"
            else:
                prompt += f"[ASSISTANT] {message['content']}\n"
        prompt += "[ASSISTANT] "
        return prompt
    
    def extract_images(self):
        """Extract images from the conversation."""
        images = []
        for message in self.messages:
            if "image" in message:
                images.append(message["image"])
        return images
