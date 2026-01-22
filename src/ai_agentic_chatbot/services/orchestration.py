from ai_agentic_chatbot.agent.intent_agent import IntentClassifier


class ChatOrchestrator:
    def __init__(self):
        self.intent_classifier = IntentClassifier()

    def process(self, question: str):
        intent_result = self.intent_classifier.classify(question)

        return {
            "intent": intent_result.intent,
            "entities": intent_result.entities,
            "confidence": intent_result.confidence,
        }
