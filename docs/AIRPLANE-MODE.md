# Airplane Mode ✈

"Airplane Mode" allows the frontend and backend to start up without relying on any internet resources. This allows
development when disconnected from the internet.

When in airplane mode:

- The backend Admin API calls and the frontend will authenticate you as testing@agency.fund. You cannot develop on
  login/logout features when airplane mode is enabled.
- testing@agency.fund will have an organization and participant type created automatically.
- The frontend may emit some errors about fonts being unavailable (these can be ignored).
- Unit tests will fail in airplane mode.

> Note: For airplane mode to work, the frontend and backend must both have been started in their non-airplane mode
> while connected to the internet.

To start the frontend in airplane mode, run:

```shell
task start-airplane
```

To start the backend in airplane mode, run:

```shell
task start-airplane
```

To run the backend tests in airplane mode, run:

```shell
task test-airplane
```
