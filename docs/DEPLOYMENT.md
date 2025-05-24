# Deployment<a name="deployment"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Deployment](#deployment)
  - [Deployment on Railway](#deployment-on-railway)
  - [Admin API](#admin-api)
  - [OIDC](#oidc)

<!-- mdformat-toc end -->

## Deployment on Railway<a name="deployment-on-railway"></a>

Our Railway deployments rely on [Dockerfile.railway](Dockerfile.railway), [railway.json](railway.json), and some
environment variables:

| Environment Variable         | Purpose                                                                                                                                                                                                    | Example                                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| XNGIN_SETTINGS               | URL of the settings to fetch. This may be a private repository.                                                                                                                                            | https://api.github.com/repos/agency-fund/xngin-settings/contents/xngin.railway.settings.json?ref=main |
| XNGIN_SETTINGS_AUTHORIZATION | The value of the "Authorization:" header sent on the request to fetch XNGIN_SETTINGS. For GitHub URLs, this token requires read access to the content of the repository and must be prefixed with `token`. | `token ghp_....`                                                                                      |
| SENTRY_DSN                   | The Sentry ingestion endpoint. If unset, Sentry will not be configured for this instance. For TAF instances, see https://agency-fund.sentry.io/settings/projects/xngin/keys/.                              | https://...@...ingest.us.sentry.io/...                                                                |
| ENVIRONMENT                  | Declares an "environment" label for the runtime environment. Used by Sentry.                                                                                                                               | xngin-main.railway.app                                                                                |

In addition, there are variables set in the Railway console corresponding to configuration values referenced by
the [xngin.railway.settings.json](https://github.com/agency-fund/xngin-settings/xngin.railway.settings.json) file in the
limited-access https://github.com/agency-fund/xngin-settings repository.

## Admin API<a name="admin-api"></a>

The Admin API supports the [xngin-dash](https://github.com/agency-fund/xngin-dash) frontend.

The Admin API is authenticated via OIDC. OIDC must be configured with additional environment variables (see below).

## OIDC<a name="oidc"></a>

| Environment Variable      | Purpose                                                                                                                                                                                                                                                           | Example                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| GOOGLE_OIDC_CLIENT_ID     | The Google-issued client ID.                                                                                                                                                                                                                                      | `2222-...apps.googleusercontent.com` |
| GOOGLE_OIDC_CLIENT_SECRET | The Google-generated client secret. Only required for PKCE.                                                                                                                                                                                                       | `G....`                              |
| GOOGLE_OIDC_REDIRECT_URI  | The URI that Google will redirect the user to after successfully authorizing. This should match the value configured in the Google Cloud console credential settings and the value embedded in the SPA. This is generally not used by the popup-style auth flows. | `http://localhost:8000/a/oidc`       |
