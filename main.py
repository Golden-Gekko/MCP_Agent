from loguru import logger

from agent import MCPCodingAgentApp

if __name__ == "__main__":
    try:
        app = MCPCodingAgentApp()
        app.run()
    except KeyboardInterrupt:
        logger.info('Выполнение прервано пользователем')
    except Exception as e:
        logger.error(e)
