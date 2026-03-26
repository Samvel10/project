from monitoring.telegram import send_telegram


class AlertManager:
    def __init__(self, telegram_token, chat_id):
        self.telegram_token = telegram_token
        self.chat_id = chat_id

    def critical(self, message):
        send_telegram(
            f"🚨 CRITICAL 🚨\n{message}",
            self.telegram_token,
            self.chat_id,
        )

    def info(self, message):
        send_telegram(
            f"ℹ️ INFO\n{message}",
            self.telegram_token,
            self.chat_id,
        )
