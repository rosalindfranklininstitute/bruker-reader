set FILE=tof
del %FILE%.py.lprof
uv run kernprof %FILE%.py -- --interactive -c config.toml
uv run python -m line_profiler -rmzt %FILE%.py.lprof > %FILE%.py.lines
echo Done. Read results in %FILE%.py.lines
