#!/usr/bin/env python3
"""Unobfuscator — CLI entry point."""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

console = Console()


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
    console.print("[green]Starting Unobfuscator daemon...[/green]")
    # Implementation added in Chunk 6


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the background daemon gracefully."""
    console.print("[yellow]Stopping daemon...[/yellow]")
    # Implementation added in Chunk 6


@cli.command()
@click.option("--doc", type=int, default=None, help="Show details for a specific document ID")
@click.pass_context
def status(ctx, doc):
    """Show processing progress across all stages."""
    console.print("[bold]Unobfuscator — Status[/bold]")
    console.print("(Daemon not yet implemented — see Chunk 6)")


@cli.command()
@click.argument("query", required=False)
@click.option("--person", default=None, help="Search by person name via Jmail people dataset")
@click.option("--date", nargs=2, default=None, metavar="FROM TO",
              help="Filter by date range (YYYY-MM-DD YYYY-MM-DD)")
@click.option("--batch", default=None, help="Target a specific release batch")
@click.option("--doc", "doc_id", type=int, default=None, help="Process a specific document ID")
@click.option("--wait", is_flag=True, help="Block until results are ready")
@click.option("--output", default=None, help="Override output directory for this run")
@click.pass_context
def search(ctx, query, person, date, batch, doc_id, wait, output):
    """Run a targeted manual search with priority +100."""
    console.print("[green]Search queued (implementation in Chunk 2+)[/green]")


@cli.group()
def config():
    """View and update configuration."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Print current configuration."""
    from core.config import load_config
    import yaml
    cfg = load_config(ctx.obj["config_path"])
    console.print(yaml.dump(cfg))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value (dot notation supported)."""
    console.print(f"[yellow]Config set {key}={value} (DB integration added in Chunk 2)[/yellow]")


if __name__ == "__main__":
    cli()
