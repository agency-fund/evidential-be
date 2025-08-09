import base64

from xngin.apiserver import settings
from xngin.apiserver.routers.admin import admin_api_types as aapi


class CredentialsUnavailableError(Exception):
    """Indicates that mutation to a datasource was requested but the datasource's credentials are unavailable."""


def api_dsn_to_settings_dwh(
    dsn: aapi.Dsn, current: settings.Dwh | None = None
) -> settings.Dwh:
    """Converts an Admin API DSN to a settings.Dwh.

    The credentials from current, if available, and of a compatible type to dsn, will be used if dsn's credentials
    are hidden.
    """
    match dsn:
        case aapi.ApiOnlyDsn():
            return settings.NoDwh()
        case aapi.BqDsn():
            if isinstance(dsn.credentials, aapi.Hidden) and isinstance(
                current, settings.BqDsn
            ):
                credentials = current.credentials
            elif isinstance(dsn.credentials, aapi.GcpServiceAccount):
                credentials = settings.GcpServiceAccountInfo(
                    content_base64=base64.standard_b64encode(
                        dsn.credentials.content.encode()
                    ).decode()
                )
            else:
                raise CredentialsUnavailableError()
            return settings.BqDsn(
                driver="bigquery",
                project_id=dsn.project_id,
                dataset_id=dsn.dataset_id,
                credentials=credentials,
            )
        case aapi.PostgresDsn() | aapi.RedshiftDsn():
            if isinstance(dsn.password, aapi.Hidden) and isinstance(
                current, settings.Dsn
            ):
                password = current.password
            elif isinstance(dsn.password, aapi.RevealedStr):
                password = dsn.password.value
            else:
                raise CredentialsUnavailableError()
            if isinstance(dsn, aapi.PostgresDsn):
                return settings.Dsn(
                    driver="postgresql+psycopg",
                    host=dsn.host,
                    port=dsn.port,
                    user=dsn.user,
                    password=password,
                    dbname=dsn.dbname,
                    sslmode=dsn.sslmode,
                    search_path=dsn.search_path,
                )
            return settings.Dsn(
                driver="postgresql+psycopg2",
                host=dsn.host,
                port=dsn.port,
                user=dsn.user,
                password=password,
                dbname=dsn.dbname,
                sslmode="verify-full",
                search_path=dsn.search_path,
            )
        case _:
            raise TypeError(type(dsn))


def settings_dwh_to_api_dsn(dwh: settings.Dwh) -> aapi.Dsn:
    """Converts a settings.Dwh to an aapi.Dsn.

    Stored credentials are never converted; they are always replaced with aapi.Hidden.
    """
    match dwh:
        case settings.NoDwh():
            return aapi.ApiOnlyDsn()
        case settings.BqDsn():
            return aapi.BqDsn(
                project_id=dwh.project_id,
                dataset_id=dwh.dataset_id,
                credentials=aapi.Hidden(),
            )
        case settings.Dsn():
            if dwh.is_redshift():
                return aapi.RedshiftDsn(
                    dbname=dwh.dbname,
                    host=dwh.host,
                    password=aapi.Hidden(),
                    port=dwh.port,
                    search_path=dwh.search_path,
                    user=dwh.user,
                )
            return aapi.PostgresDsn(
                dbname=dwh.dbname,
                host=dwh.host,
                password=aapi.Hidden(),
                port=dwh.port,
                search_path=dwh.search_path,
                sslmode=dwh.sslmode,
                user=dwh.user,
            )
        case _:
            raise TypeError(dwh)
