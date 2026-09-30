"""Paper trading en modo demostración (sin conexión) sobre datos sintéticos."""

from modulos.trading import repositorio as repo
from modulos.trading.experimento import ejecutar_experimento
from modulos.trading.paper import paper_trading
from tests.test_experimento import CAMBIOS


def test_paper_sintetico():
    exp_id = ejecutar_experimento(CAMBIOS, "paper", lanzar_agentes=False)
    est_id = repo.finalistas(exp_id)[0]
    pid = paper_trading.registrar(est_id, fuente="sintetico")
    estado = paper_trading.actualizar(pid, enviar_alertas=False)
    assert "equity" in estado and estado["velas_seguidas"] > 0
    ops = paper_trading.operaciones(pid)
    # Actualizar otra vez no duplica operaciones
    paper_trading.actualizar(pid, enviar_alertas=False)
    assert len(paper_trading.operaciones(pid)) == len(ops)
    comp = paper_trading.comparar_con_backtest(pid)
    assert list(comp.columns) == ["Backtest (test)", "Paper (en vivo)"]
    paper_trading.pausar(pid, False)
    assert paper_trading.actualizar_todas() == {}
    paper_trading.borrar(pid)
    assert paper_trading.listar().query("id == @pid").empty
