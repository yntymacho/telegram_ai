from pathlib import Path
from dotenv import load_dotenv
import sys

# Load environment variables before importing other modules
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path, override=True)

from loguru import logger
from bot import SalesBot

def setup_logging():
    """Configure logging settings"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "bot.log",
        rotation="500 MB",
        retention="10 days",
        compression="zip"
    )

def main():
    """Main entry point"""
    try:
        setup_logging()
        logger.info("Starting Sales Assistant Bot")
        
        bot = SalesBot()
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Bot crashed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
