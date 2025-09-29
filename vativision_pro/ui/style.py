"""Application-wide Qt stylesheet helpers."""

from __future__ import annotations

from ..config import DARK_BG, DARK_CARD, DARK_ELEV, ACCENT, TEXT

def style():
    return f"""
    QWidget {{ background-color:{DARK_BG}; color:{TEXT}; font-family:Segoe UI, Inter, Roboto, Arial; font-size:14px; }}
    QGroupBox {{ background-color:{DARK_CARD}; border:1px solid #3a3c41; border-radius:10px; margin-top:16px; padding:12px; }}
    QPushButton {{ background-color:{ACCENT}; color:white; border:none; border-radius:8px; padding:8px 12px; font-weight:600; }}
    QLabel#status {{ background-color:{DARK_CARD}; border:1px solid #3a3c41; border-radius:8px; padding:8px; }}
    QPlainTextEdit {{ background-color:{DARK_ELEV}; border:1px solid #3a3c41; border-radius:8px; }}
    QComboBox, QLineEdit, QSlider {{ background-color:{DARK_ELEV}; border:1px solid #3a3c41; border-radius:8px; padding:6px; }}
    QCheckBox {{ background:transparent; }}
    """
