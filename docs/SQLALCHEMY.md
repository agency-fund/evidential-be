# SQLAlchemy Tips

## async considerations

https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#asyncio-orm-avoid-lazyloads

## expire_on_commit

The SQLAlchemy sessions connecting our application to the database are configured with the `expire_on_commit` setting
set to False. This changes the default behavior. We do this because it allows us to write simpler code, with
fewer variables, and avoids surprising database queries.

The default behavior is:

```python
# expire_on_commit defaults to True
session = Session(engine)

stmt = select(Experiment).where(Experiment.id == "exp_123")
experiment = session.execute(stmt).scalar_one_or_none()
print(f"Experiment name: {experiment.name}")  # Loads the name from the database
experiment.name = "Updated Experiment"
session.commit()

# After commit, the experiment object is expired and will automatically be refreshed
print(f"Experiment name after commit: {experiment.name}")  # New DB query happens here!
```

By setting it to false, the behavior changes. Objects after the commit are no-longer invalidated. This means that if
you are reading any database-generated values from newly inserted or updated objects, you must explicitly refresh the
object. Here's how:

```python
session = Session(engine, expire_on_commit=False)

stmt = select(Experiment).where(Experiment.id == "exp_123")
experiment = session.execute(stmt).scalar_one_or_none()
print(f"Experiment name: {experiment.name}")  # Loads the name from the database
experiment.name = "Updated Experiment"
session.commit()

# After commit, the experiment object is NOT expired
# No new query is made - we use the in-memory state
print(f"Experiment name after commit: {experiment.name}")  # No DB query!

# However, if the database generates values on commit (like updated_at timestamps),
# you won't see those changes unless you explicitly refresh:
session.refresh(experiment)
print(f"Updated timestamp: {experiment.updated_at}")  # Now has the latest DB values
```
