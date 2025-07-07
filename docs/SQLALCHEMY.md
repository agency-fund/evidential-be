# SQLAlchemy Tips

## Preloading & Async

To
avoid [Implicit I/O](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#preventing-implicit-io-when-using-asyncsession)
when interacting with the database, we have adopted some conventions:

1. When reading a `relationship()` attribute of a table model, you can use
   `select(tables.T).options(selectinload(tables.T.rel))`
   on the query to avoid accidental implicit I/O later. Some of the database helper methods have a `preload=` attribute
   to make this more ergonomic.
1. Our base table definition class includes [
   `AsyncAttrs`](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#sqlalchemy.ext.asyncio.AsyncAttrs)
   so that you may read attributes using async I/O as needed.

Note that visual inspection may not reveal all cases of implicit I/O due to SQLAlchemy's ORM session cache. If you
are unsure if you are triggering implicit I/O, set `lazy="raise"` on the relationship field and an exception will be
raised when it occurs.

To maintain async correctness, implicit I/O on the application database should be considered a bug.

## Interactions with Customer Data Warehouses

We support multiple data warehouse backends but not all of them have async drivers for SQLAlchemy. Thus we
have adopted a convention of running all queries against customer databases in a thread. This allows our request
threads to remain non-blocking. See [DwhSession](../src/xngin/apiserver/dwh/dwh_session.py) for additional details.

## expire_on_commit

The SQLAlchemy sessions connecting our application to the database are configured with the `expire_on_commit` setting
set to False. This changes the default behavior. We do this because it allows us to write simpler code, with
fewer variables, and avoids surprising database queries.

The default behavior is:

```python
# expire_on_commit defaults to True
async with AsyncSession(engine) as session:
    stmt = select(Experiment).where(Experiment.id == "exp_123")
    result = await session.execute(stmt)
    experiment = result.scalar_one_or_none()
    print(f"Experiment name: {experiment.name}")
    experiment.name = "Updated Experiment"
    await session.commit()

    # After commit, the experiment object is expired and will automatically be refreshed
    print(
        f"Experiment name after commit: {experiment.name}"
    )  # New DB query happens here!
```

By setting it to false, the behavior changes. Objects after the commit are no-longer invalidated. This means that if
you are reading any database-generated values from newly inserted or updated objects, you must explicitly refresh the
object. Here's how:

```python
async with AsyncSession(engine, expire_on_commit=False) as session:
    stmt = select(Experiment).where(Experiment.id == "exp_123")
    result = await session.execute(stmt)
    experiment = result.scalar_one_or_none()
    print(f"Experiment name: {experiment.name}")
    experiment.name = "Updated Experiment"
    await session.commit()

    # After commit, the experiment object is NOT expired
    # No new query is made - we use the in-memory state
    print(f"Experiment name after commit: {experiment.name}")  # No DB query!

    # However, if the database generates values on commit (like updated_at timestamps),
    # you won't see those changes unless you explicitly refresh:
    await session.refresh(experiment)
    print(f"Updated timestamp: {experiment.updated_at}")  # Now has the latest DB values
```
