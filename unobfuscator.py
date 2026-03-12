#!/usr/bin/env python3
"""Unobfuscator — CLI entry point."""

import json
import logging
import os
import signal
import time
import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional
from core.db import (
    init_db, get_connection, get_known_batch_ids, insert_release_batch,
    get_pending_pdf_documents, get_documents_by_ids, reset_group_merged,
    set_config, get_config
)
from core.config import load_config, get as cfg_get
from core.api import fetch_release_batches, fetch_person_document_ids
from core.queue import enqueue, get_queue_stats, dequeue, mark_done
from stages.indexer import run_indexer_batch
from stages.matcher import (
    run_phase0_email_fastpath, run_phase2_lsh_candidates,
    run_phase3_verify_and_group
)
from stages.merger import run_merger
from stages.pdf_processor import process_pdf_for_document
from stages.output_generator import run_output_generator

console = Console()
logger = logging.getLogger(__name__)

PID_FILE = ".unobfuscator.pid"
_shutdown_requested = False
_daemon_conn = None  # set during start, used by _set_activity


def _set_activity(msg: str) -> None:
    """Write current daemon activity to DB so 'status' can display it."""
    if _daemon_conn is None:
        return
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    set_config(_daemon_conn, "daemon_activity", f"{ts} | {msg}")
    _daemon_conn.commit()


def _write_pid() -> None:
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _read_pid() -> Optional[int]:
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def _remove_pid() -> None:
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def _run_one_cycle(conn, cfg: dict) -> None:
    """Run one full pass of all 5 stages using only core.db helpers — no inline SQL."""
    global _shutdown_requested
    markers = cfg_get(cfg, "redaction_markers", default=[])
    min_overlap = cfg_get(cfg, "matching.min_overlap_chars", default=200)
    threshold = cfg_get(cfg, "matching.similarity_threshold", default=0.70)
    min_headers = cfg_get(cfg, "matching.email_header_min_matches", default=2)
    output_dir = cfg_get(cfg, "output_dir", default="./output")

    if _shutdown_requested:
        return

    batch_ids = get_known_batch_ids(conn)
    for i, batch_id in enumerate(batch_ids, 1):
        if _shutdown_requested:
            return
        _set_activity(f"Stage 1 Indexer: batch {i}/{len(batch_ids)} ({batch_id})")
        run_indexer_batch(conn, batch_id=batch_id, redaction_markers=markers)

    if _shutdown_requested:
        return

    _set_activity("Stage 2 Matcher: email fastpath")
    run_phase0_email_fastpath(conn, min_header_matches=min_headers)
    _set_activity("Stage 2 Matcher: LSH candidates")
    candidates = run_phase2_lsh_candidates(conn, threshold=threshold)
    _set_activity(f"Stage 2 Matcher: verifying {len(candidates)} candidates")
    run_phase3_verify_and_group(conn, candidates, redaction_markers=markers,
                                min_overlap_chars=min_overlap)

    if _shutdown_requested:
        return

    _set_activity("Stage 3 Merger: merging groups")
    run_merger(conn, redaction_markers=markers)

    # Process any pending merge queue jobs (e.g., from soft-redaction discoveries).
    merge_job = dequeue(conn, stage="merge")
    while merge_job:
        if _shutdown_requested:
            break
        payload = json.loads(merge_job["payload"])
        group_id = payload.get("group_id")
        if group_id is not None:
            # Reset merged flag so run_merger picks it up again.
            reset_group_merged(conn, group_id)
        else:
            console.print("[yellow]Merge job with no group_id skipped.[/yellow]")
        mark_done(conn, merge_job["job_id"])
        merge_job = dequeue(conn, stage="merge")

    # Re-run merger to process any groups that were just reset.
    if not _shutdown_requested:
        run_merger(conn, redaction_markers=markers)

    if not _shutdown_requested:
        pdf_limit = cfg_get(cfg, "workers.pdf", default=2)
        pdf_docs = get_pending_pdf_documents(conn, limit=pdf_limit)
        for i, pdf_doc in enumerate(pdf_docs, 1):
            if _shutdown_requested:
                break
            _set_activity(f"Stage 4 PDF: processing {i}/{len(pdf_docs)}")
            process_pdf_for_document(conn, doc_id=pdf_doc["id"])

    if not _shutdown_requested:
        _set_activity("Stage 5 Output: generating files")
        run_output_generator(conn, output_dir=output_dir, redaction_markers=markers)

    # Mark pending index jobs done — the cycle already processed all DB state.
    index_job = dequeue(conn, stage="index")
    while index_job:
        if _shutdown_requested:
            break
        mark_done(conn, index_job["job_id"])
        index_job = dequeue(conn, stage="index")

    conn.commit()


def _poll_for_new_batches(conn, cfg: dict) -> bool:
    """Check Jmail for new release batches and queue indexing for any new ones.

    Returns True on success, False if the fetch failed after retries.
    """
    known = get_known_batch_ids(conn)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            current = set(fetch_release_batches())
            break
        except Exception as e:
            if attempt < max_retries:
                wait = 30 * attempt
                logger.warning(
                    "Failed to fetch release batches (attempt %d/%d): %s — "
                    "retrying in %ds", attempt, max_retries, e, wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Failed to fetch release batches after %d attempts: %s",
                    max_retries, e,
                )
                return False
    else:
        return False  # shouldn't reach here, but be safe

    for batch_id in current - known:
        insert_release_batch(conn, batch_id)
        enqueue(conn, stage="index", payload={"batch_id": batch_id})
    conn.commit()
    return True


@click.group()
@click.option("--config", default="config.yaml", show_default=True,
              help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """Unobfuscator: cross-reference and unredact Epstein archive documents."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.pass_context
def start(ctx):
    """Start the background daemon (all 5 stages)."""
    global _shutdown_requested
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    poll_interval = cfg_get(cfg, "polling.interval_minutes", default=60) * 60

    init_db(db_path)
    conn = get_connection(db_path)
    _shutdown_requested = False

    log_path = cfg_get(cfg, "log_path", default="./data/unobfuscator.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    global _daemon_conn
    _daemon_conn = conn
    _write_pid()
    console.print(f"[green]Daemon started (PID {os.getpid()})[/green]")

    def _handle_signal(sig, frame):
        global _shutdown_requested
        _shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    retry_interval = 5 * 60  # 5 min retry on poll failure
    try:
        _set_activity("Polling for new batches (initial)")
        _poll_for_new_batches(conn, cfg)
        while not _shutdown_requested:
            _run_one_cycle(conn, cfg)
            stats = get_queue_stats(conn)
            if stats.get("pending", 0) == 0 and not _shutdown_requested:
                _set_activity("Polling for new batches")
                poll_ok = _poll_for_new_batches(conn, cfg)
                wait = poll_interval if poll_ok else retry_interval
                from datetime import datetime, timezone, timedelta
                wake = datetime.now(timezone.utc) + timedelta(seconds=wait)
                reason = "" if poll_ok else " (poll failed, retrying sooner)"
                _set_activity(
                    f"Idle — sleeping {wait // 60:.0f}m until "
                    f"{wake.astimezone().strftime('%H:%M')}{reason}"
                )
                console.print(
                    f"[dim]All tasks complete. Checking for updates in "
                    f"{wait // 60:.0f} min...[/dim]"
                )
                for _ in range(int(wait)):
                    if _shutdown_requested:
                        break
                    time.sleep(1)
    finally:
        _set_activity("Stopped")
        _daemon_conn = None
        _remove_pid()
        console.print("[yellow]Daemon stopped.[/yellow]")


@cli.command()
@click.option("--force", is_flag=True, help="Force-kill if graceful stop fails within 10s")
@click.pass_context
def stop(ctx, force):
    """Stop the background daemon gracefully (or forcefully with --force)."""
    pid = _read_pid()
    if pid is None:
        console.print("[red]No daemon running.[/red]")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(
            f"[yellow]Stop signal sent to PID {pid}.[/yellow]"
        )
    except ProcessLookupError:
        console.print("[red]Daemon not found — cleaning up.[/red]")
        _remove_pid()
        return

    if not force:
        return

    # Wait up to 10s for graceful shutdown, then SIGKILL.
    for _ in range(20):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)  # check if alive
        except ProcessLookupError:
            console.print("[green]Daemon stopped.[/green]")
            _remove_pid()
            return

    console.print(f"[red]Daemon still alive — force-killing PID {pid}.[/red]")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_pid()
    console.print("[green]Done.[/green]")


@cli.command()
@click.option("--doc", type=str, default=None, help="Show details for a specific document ID")
@click.pass_context
def status(ctx, doc):
    """Show processing progress across all stages."""
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    if not os.path.exists(db_path):
        console.print("[red]Database not found. Run 'unobfuscator start' first.[/red]")
        return
    conn = get_connection(db_path)

    if doc:
        rows = get_documents_by_ids(conn, [doc])
        console.print(rows[0] if rows else f"[red]Document {doc} not found.[/red]")
        return

    pid = _read_pid()
    daemon_label = f"running (PID {pid})" if pid else "stopped"
    activity = get_config(conn, "daemon_activity", default="unknown")

    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    indexed = conn.execute("SELECT COUNT(*) FROM documents WHERE text_processed=1").fetchone()[0]
    fps = conn.execute("SELECT COUNT(*) FROM document_fingerprints").fetchone()[0]
    groups = conn.execute("SELECT COUNT(*) FROM merge_results").fetchone()[0]
    pdf_done = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_processed=1").fetchone()[0]
    pdf_total = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_url IS NOT NULL"
    ).fetchone()[0]
    output_count = conn.execute(
        "SELECT COUNT(*) FROM merge_results WHERE output_generated=1"
    ).fetchone()[0]
    recovered = conn.execute(
        "SELECT COALESCE(SUM(recovered_count),0) FROM merge_results"
    ).fetchone()[0]
    output_dir = cfg_get(cfg, "output_dir", default="./output")

    t = Table(title="Unobfuscator — Status", show_header=False)
    t.add_column("", style="bold")
    t.add_column("")
    t.add_row("Daemon", daemon_label)
    if pid:
        t.add_row("Activity", activity)
    t.add_row("Stage 1 Indexer", f"{indexed:,} / {total:,} docs")
    t.add_row("Stage 2 Matcher", f"{fps:,} fingerprints built")
    t.add_row("Stage 3 Merger", f"{groups:,} groups merged")
    t.add_row("Stage 4 PDF Processor", f"{pdf_done:,} / {pdf_total:,} PDFs done")
    t.add_row("Stage 5 Output", f"{output_count:,} files written")
    t.add_row("Recovered redactions", f"{recovered:,} total")
    t.add_row("Output directory", output_dir)
    console.print(t)


@cli.command()
@click.argument("query", required=False)
@click.option("--person", default=None, help="Search by person name via Jmail people dataset")
@click.option("--date", nargs=2, default=None, metavar="FROM TO",
              help="Filter by date range (YYYY-MM-DD YYYY-MM-DD)")
@click.option("--batch", default=None, help="Target a specific release batch")
@click.option("--doc", "doc_id", type=str, default=None, help="Process a specific document ID")
@click.option("--wait", is_flag=True, help="Block until results are ready")
@click.option("--output", default=None, help="Override output directory for this run")
@click.pass_context
def search(ctx, query, person, date, batch, doc_id, wait, output):
    """Run a targeted manual search with priority +100."""
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    init_db(db_path)
    conn = get_connection(db_path)

    if person:
        console.print(f"[dim]Looking up '{person}' in Jmail people dataset...[/dim]")
        try:
            doc_ids = fetch_person_document_ids(person)
            if doc_ids:
                for did in doc_ids:
                    enqueue(conn, stage="index",
                            payload={"doc_id": did, "output_dir": output},
                            priority=100)
                console.print(
                    f"[green]Queued {len(doc_ids)} documents for '{person}'[/green]"
                )
            else:
                console.print(f"[yellow]No documents found for '{person}'[/yellow]")
        except Exception as e:
            console.print(f"[red]People lookup failed: {e}[/red]")
    else:
        payload: dict = {}
        if query:
            payload["query"] = query
        if batch:
            payload["batch"] = batch
        if doc_id:
            payload["doc_id"] = doc_id
        if output:
            payload["output_dir"] = output
        enqueue(conn, stage="index", payload=payload, priority=100)
        console.print(f"[green]Search queued: {payload}[/green]")

    conn.commit()

    if wait:
        console.print("[dim]Waiting for daemon to process...[/dim]")
        deadline = time.time() + 300  # 5-minute timeout
        while time.time() < deadline:
            stats = get_queue_stats(conn)
            if stats.get("pending", 0) == 0 and stats.get("running", 0) == 0:
                break
            time.sleep(2)
        else:
            console.print("[yellow]Timed out waiting for daemon.[/yellow]")
        ctx.invoke(status)


@cli.command()
@click.option("-n", "--lines", default=50, show_default=True, help="Number of lines to show")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f)")
@click.pass_context
def log(ctx, lines, follow):
    """Show the daemon log."""
    cfg = load_config(ctx.obj["config_path"])
    log_path = cfg_get(cfg, "log_path", default="./data/unobfuscator.log")
    if not os.path.exists(log_path):
        console.print("[red]No log file found.[/red]")
        return
    import subprocess
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(log_path)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


@cli.group()
def config():
    """View and update configuration."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Print current configuration."""
    import yaml
    cfg = load_config(ctx.obj["config_path"])
    console.print(yaml.dump(cfg))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value (dot notation supported)."""
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    init_db(db_path)
    conn = get_connection(db_path)
    set_config(conn, key, value)
    conn.commit()
    conn.close()
    console.print(f"[green]Config set: {key} = {value}[/green]")


if __name__ == "__main__":
    cli()
