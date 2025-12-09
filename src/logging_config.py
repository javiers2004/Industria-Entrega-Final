"""
Configuracion centralizada de logging para el proyecto EAF.
"""
import logging
from typing import Optional


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configura y retorna un logger para el modulo especificado.

    Args:
        name: Nombre del logger (usa __name__ del modulo que lo llama).
              Si es None, configura el logger raiz.
        level: Nivel de logging (default: INFO)

    Returns:
        Logger configurado

    Example:
        >>> from src.logging_config import setup_logging
        >>> logger = setup_logging(__name__)
        >>> logger.info("Mensaje informativo")
    """
    # Configurar formato solo si no hay handlers configurados
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger ya configurado por nombre.

    Args:
        name: Nombre del logger (tipicamente __name__)

    Returns:
        Logger configurado
    """
    return setup_logging(name)
