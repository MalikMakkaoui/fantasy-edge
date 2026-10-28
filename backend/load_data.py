from pathlib import Path
import nflreadpy as nfl
import polars as pl

POSITIONS = ["QB", "RB", "WR", "TE"]
SEASONS = [2022, 2023, 2024, 2025]

MIN_SNAP_SHARE = 0.50

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_nfl_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    #Download weekly nfl stats and snap counts

    stats = nfl.load_player_stats(
        seasons=SEASONS,
        summary_level="week"
    )

    snaps = nfl.load_snap_counts(
        seasons = SEASONS
    )


    # keep only fantasy-relevant positions include regular season and post season

    stats = stats.filter(
        pl.col("position").is_in(POSITIONS)
        & pl.col("season_type").is_in(["REG", "POST"])
    )

    return stats,snaps

# Function for Matching fantasy stats with offensive snap percentages
def combine_player_data(
    stats:pl.DataFrame,
    snaps: pl.DataFrame
) -> pl.DataFrame:

    print("Stats columns:", stats.columns)
    print("Snaps columns:", snaps.columns)

    # keep the columns we need from the snap dataset
    snaps = snaps.select([
        "season",
        "week",
        "game_type",
        "team",
        "player",
        "offense_pct"
    ])

    # rename fields so both datasets have matching names
    snaps = snaps.rename({
        "game_type": "season_type",
        "player": "player_display_name",
    })

    # Normalize player names for an initial join
    stats = stats.with_columns(
        pl.col("player_display_name")
        .str.to_lowercase()
        .str.strip_chars()
        .alias("match_name")
    )

    snaps = snaps.with_columns(
        pl.col("player_display_name")
        .str.to_lowercase()
        .str.strip_chars()
        .alias("match_name")
    )

    #Convert percentages to a consistent 0-1 scale

    snaps = snaps.with_columns(
        pl.when(pl.col("offense_pct") > 1)
        .then(pl.col("offense_pct") / 100)
        .otherwise(pl.col("offense_pct"))
        .alias("snap_share")
    )

    join_keys = [
        "season",
        "week",
        "season_type",
        "team",
        "match_name"
    ]

    duplicate_keys = (
        snaps.group_by(join_keys)
        .len()
        .filter(pl.col("len") > 1)
    )

    if not duplicate_keys.is_empty():
        raise ValueError(
            "Duplicate snap-count matches found."
            "Review player identifiers before joining"
        )
    
    # Combine the two datasets
    combined = stats.join(
        snaps.select(join_keys + ["snap_share"]),
        on=join_keys,
        how="left",
        validate="m:1"
    )

    # Report rows that couldn't be matched
    missing = combined.filter(
        pl.col("snap_share").is_null()
    )

    print(f"Unmatched player-game records: {missing.height}")

    # Apply our > 50% participation rule
    qualified = combined.filter(
        pl.col("snap_share") > MIN_SNAP_SHARE
    )

    return qualified

def calculate_season_ppg(stats: pl.DataFrame) -> pl.DataFrame:
# calculate PPG using qualifying games only

    results = (
        stats.group_by([
            "player_id",
            "player_display_name",
            "position",
            "season"
        ])
        .agg([
            pl.col("fantasy_points_ppr")
            .sum()
            .alias("total_ppr"),

            pl.col("fantasy_points_ppr")
            .count()
            .alias("qualifying_games"),

            pl.col("snap_share")
            .mean()
            .round(3)
            .alias("avg_snap_share")
        ])
        .filter(
            pl.col("qualifying_games") >= 5
        )
        .with_columns(
            (
                pl.col("total_ppr")
                /pl.col("qualifying_games")
            )
            .round(2)
            .alias("ppg")
        )
        .sort(
            ["season", "ppg"],
            descending=[True,True]
        )
    )
    return results

def main() -> None:

    # Run our entire data pipeline.

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading NFL data...")
    stats,snaps = load_nfl_data()

    print("Matching snap counts...")
    qualifed_stats = combine_player_data(stats,snaps)

    print("calculating adjusted fantasy PPG...")
    results = calculate_season_ppg(qualifed_stats)

    print(results.head(20))

    results.write_csv(
        DATA_DIR / "player_seasons.csv"
    )

    print("data saved successfully")

if __name__ == "__main__":
    main()


