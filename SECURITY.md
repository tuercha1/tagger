# Security

Do not commit local runtime files:

- `config.json`
- `translations.db`
- `.venv/`
- log files
- API keys or private endpoints

Use `config.example.json` as the public template. If a key has ever been committed
or shared, revoke it from the provider console and create a new one.
