import asyncio
from typing import List, Dict, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from loguru import logger
from datetime import datetime

from config import config
from vector_store import VectorStore
from google_sheets import GoogleSheetsLoader

class SalesBot:
    def __init__(self):
        """Initialize bot with vector store and API clients"""
        try:
            # Validate configuration
            config.validate_paths()
            config.validate_credentials()
            
            # Initialize components
            self.vector_store = VectorStore()
            self.sheets_loader = GoogleSheetsLoader()
            self.mistral_client = MistralClient(api_key=config.MISTRAL_API_KEY)
            
            # Load initial data
            self._load_qa_data()
            
            logger.info("Sales bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sales bot: {str(e)}")
            raise

    def _load_qa_data(self) -> None:
        """Load QA data from Google Sheets to vector store"""
        try:
            df = self.sheets_loader.load_from_sheet()
            self.vector_store.load_data(df)
        except Exception as e:
            logger.error(f"Failed to load QA data: {str(e)}")
            raise

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        welcome_message = (
            "👋 Welcome to the Sales Assistant!\n\n"
            "I can help you with questions about our products and services. "
            "Feel free to ask anything!"
        )
        await update.message.reply_text(welcome_message)

    async def refresh(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /refresh command to reload QA data"""
        try:
            self._load_qa_data()
            await update.message.reply_text("✅ Successfully refreshed QA data!")
        except Exception as e:
            error_message = "❌ Failed to refresh QA data. Please try again later."
            logger.error(f"Refresh failed: {str(e)}")
            await update.message.reply_text(error_message)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user messages with RAG pipeline"""
        try:
            # Get relevant context from vector store
            user_query = update.message.text
            relevant_docs = self.vector_store.search(user_query)
            
            if not relevant_docs:
                await update.message.reply_text(
                    "I couldn't find any relevant information. "
                    "Please try rephrasing your question or contact support."
                )
                return
            
            # Format context for the model
            context = "\n\n".join([
                f"Q: {doc['question']}\nA: {doc['answer']}"
                for doc in relevant_docs
            ])
            
            # Generate response using Mistral
            system_prompt = (
                "You are a helpful sales assistant. Use the following Q&A pairs as context "
                "to answer the user's question. If you're not sure about something, "
                "say so and offer to connect them with human support.\n\n"
                f"Context:\n{context}"
            )
            
            response = self.mistral_client.chat(
                model=config.MISTRAL_MODEL,
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_query)
                ],
                temperature=config.TEMPERATURE
            )
            
            await update.message.reply_text(response.choices[0].message.content)
            
        except Exception as e:
            error_message = (
                "I apologize, but I encountered an error processing your request. "
                "Please try again later or contact support if the issue persists."
            )
            logger.error(f"Message handling failed: {str(e)}")
            await update.message.reply_text(error_message)

    async def refresh_data(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Periodic task to refresh QA data"""
        try:
            self._load_qa_data()
            logger.info(f"Scheduled data refresh completed at {datetime.now()}")
        except Exception as e:
            logger.error(f"Scheduled data refresh failed: {str(e)}")

    def run(self) -> None:
        """Start the bot"""
        try:
            # Initialize application
            app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
            
            # Add handlers
            app.add_handler(CommandHandler("start", self.start))
            app.add_handler(CommandHandler("refresh", self.refresh))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            # Schedule periodic data refresh
            app.job_queue.run_repeating(
                self.refresh_data,
                interval=86400,  # 24 hours
                first=10
            )
            
            # Start polling
            logger.info("Starting bot polling...")
            app.run_polling()
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}")
            raise
