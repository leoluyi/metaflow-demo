import smtplib
from email.message import EmailMessage
from typing import List


class EmailAlerts:
    def __init__(self, smtp_config: dict):
        self.config = smtp_config

    def send_alert(self, alerts: List[str], recipients: List[str]):
        msg = EmailMessage()
        msg.set_content("\n".join(alerts))
        msg["Subject"] = "ML Pipeline Alerts"
        msg["From"] = self.config["sender"]
        msg["To"] = ", ".join(recipients)

        with smtplib.SMTP(self.config["host"], self.config["port"]) as server:
            if self.config.get("use_tls", False):
                server.starttls()
            if "username" in self.config:
                server.login(self.config["username"], self.config["password"])
            server.send_message(msg)
