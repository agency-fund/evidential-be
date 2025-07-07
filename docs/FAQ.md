# FAQ<a name="faq"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [FAQ](#faq)
  - [How do I hide the generated SQL from the logs or console output?](#how-do-i-hide-the-generated-sql-from-the-logs-or-console-output)
  - [psycopg2 module does not install correctly.](#psycopg2-module-does-not-install-correctly)

<!-- mdformat-toc end -->

## How do I hide the generated SQL from the logs or console output?<a name="how-do-i-hide-the-generated-sql-from-the-logs-or-console-output"></a>

See the environment variables in [Taskfile.yml](../Taskfile.yml) and customlogging.py for documentation on how to
control logging.

## psycopg2 module does not install correctly.<a name="psycopg2-module-does-not-install-correctly"></a>

You might see this error:

> Error: pg_config executable not found.
>
> pg_config is required to build psycopg2 from source.

The fix will depend on your specific environment.

### Linux

1. If on Linux, try: `sudo apt install -y libpq-dev` and then re-install dependencies.
1. See https://www.psycopg.org/docs/install.html.
1. See https://www.psycopg.org/docs/faq.html.

### OSX<a name="osx"></a>

Run `brew install postgresql@17`.
