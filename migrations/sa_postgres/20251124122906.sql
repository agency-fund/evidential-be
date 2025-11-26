with pos as (
    select
        *,
        -- a table scan is most likely to maintain insertion order numbering
        row_number() over () as total_r
    from arms
),

rel_pos as (
    -- now we can number items relative to their own experiment_id
    select
        *,
        row_number() over (partition by experiment_id order by total_r asc) as r
    from pos
),

joined_pos as (
    select
        r.r,
        r.total_r,
        r.name as arm_name,
        r.description,
        e.name as exp,
        e.id as exp_id,
        o.name as org,
        o.id as org_id,
        r.created_at,
        r.id as arm_id
    from rel_pos as r
    inner join organizations as o on (r.organization_id = o.id)
    inner join experiments as e on (r.experiment_id = e.id)
)

-- manually inspect the pending updates
-- select * from joined_pos order by org, created_at, r;
update arms set position = (
    select joined_pos.r from joined_pos
    where joined_pos.arm_id = arms.id
);
