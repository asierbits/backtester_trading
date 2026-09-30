"""Registro del módulo Trading en el menú lateral."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

NOMBRE = "Trading"
ORDEN = 10
DESCRIPCION = "Laboratorio de estrategias con dinero ficticio: backtesting honesto y detector de humo."

_P = Path(__file__).resolve().parent / "paginas"


def paginas() -> list:
    return [
        st.Page(_P / "inicio.py", title="Inicio", icon=":material/home:", url_path="trading"),
        st.Page(_P / "datos.py", title="Datos", icon=":material/database:", url_path="datos"),
        st.Page(_P / "nuevo_experimento.py", title="Nuevo experimento", icon=":material/science:", url_path="nuevo"),
        st.Page(_P / "ranking.py", title="Ranking", icon=":material/leaderboard:", url_path="ranking"),
        st.Page(_P / "detalle.py", title="Detalle de estrategia", icon=":material/query_stats:", url_path="detalle"),
        st.Page(_P / "humo.py", title="Laboratorio de humo", icon=":material/local_fire_department:", url_path="humo"),
        st.Page(_P / "paper.py", title="Paper trading", icon=":material/monitoring:", url_path="paper"),
        st.Page(_P / "informes.py", title="Informes", icon=":material/description:", url_path="informes"),
        st.Page(_P / "ajustes.py", title="Ajustes", icon=":material/settings:", url_path="ajustes"),
    ]
