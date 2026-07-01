# ─── TAB 2: Leaderboard ───────────────────────────────────────────────────────

def tab_leaderboard(s_df: pd.DataFrame, d_df: pd.DataFrame, filters: dict) -> None:
    if s_df.empty:
        st.warning("No data matches the current filters.")
        return

    sel_groups  = filters.get("sel_groups", ALL_GROUPS)
    all_m_keys  = [m.key for g in sel_groups for m in METRICS_BY_GROUP.get(g, [])
                   if m.key in s_df.columns]
    avail_metrics = [k for k in all_m_keys if s_df[k].notna().any()]

    # ── Horizontal Options Row (Above the Table) ──
    section_label("Leaderboard Options")
    opt_c1, opt_c2, opt_c3 = st.columns([2, 1, 1])
    
    with opt_c1:
        ranking_metrics = st.multiselect(
            "Ranking metrics",
            avail_metrics,
            default=[k for k in DEFAULT_PRIMARY_METRICS if k in avail_metrics],
            format_func=metric_label,
        )
    with opt_c2:
        aggregate = st.selectbox(
            "Aggregate across datasets by",
            ["mean", "median", "best-run", "worst-run"],
        )
    with opt_c3:
        n_models = s_df["model_key"].nunique()
        top_n = st.slider("Top N models", min_value=1, max_value=n_models, value=min(10, n_models))

    if not ranking_metrics:
        st.info("Select at least one ranking metric to view the leaderboard.")
        return

    # Aggregation logic
    if aggregate == "best-run":
        agg_fn = {k: ("min" if metric_direction(k) == "lower" else "max") for k in avail_metrics}
    elif aggregate == "worst-run":
        agg_fn = {k: ("max" if metric_direction(k) == "lower" else "min") for k in avail_metrics}
    else:
        agg_fn = aggregate

    agg_df = _agg_summary(s_df, avail_metrics, agg_fn=agg_fn)
    ranked = _rank_df(agg_df, ranking_metrics).sort_values("composite_score", ascending=False).head(top_n).reset_index(drop=True)

    # ── Full-Width Leaderboard Table ──
    section_label("Leaderboard")
    
    display_df = ranked[["display_name", "source", "categories", "composite_score"] + ranking_metrics].copy()
    display_df.insert(0, "Rank", [medal(i+1) for i in range(len(display_df))])
    display_df["categories"] = display_df["categories"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    display_df["composite_score"] = display_df["composite_score"] * 100

    rename_map = {
        "display_name": "Model",
        "source": "Source",
        "categories": "Categories",
        "composite_score": "Composite Score (%)"
    }
    for m in ranking_metrics:
        dir_arrow = "↓" if metric_direction(m) == "lower" else "↑"
        rename_map[m] = f"{metric_label(m)} {dir_arrow}"
        
    display_df = display_df.rename(columns=rename_map)

    def highlight_podium(s, direction):
        ranks = s.rank(method='min', ascending=(direction == 'lower'))
        styles = []
        for r in ranks:
            if pd.isna(r):
                styles.append('')
            elif r == 1:
                styles.append(f'background-color: rgba(217, 119, 6, 0.15); font-weight: bold; color: {PALETTE["warn"]}')
            elif r == 2:
                styles.append(f'background-color: rgba(100, 116, 139, 0.15); font-weight: bold; color: {PALETTE["text_muted"]}')
            elif r == 3:
                styles.append(f'background-color: rgba(140, 99, 56, 0.15); font-weight: bold; color: #8c6338')
            else:
                styles.append('')
        return styles

    styler = display_df.style
    styler = styler.apply(highlight_podium, direction="higher", subset=["Composite Score (%)"])
    
    for m in ranking_metrics:
        col_name = rename_map[m]
        styler = styler.apply(highlight_podium, direction=metric_direction(m), subset=[col_name])
        
    format_dict = {"Composite Score (%)": "{:.1f}%"}
    for m in ranking_metrics:
        fmt = get(m).format if get(m) else ".4f"
        format_dict[rename_map[m]] = f"{{:{fmt}}}"
        
    styler = styler.format(format_dict, na_rep="—")

    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Charts ──
    global_color_map = _model_color_map(agg_df["display_name"].tolist())

    c_left, c_right = st.columns(2)
    
    with c_left:
        section_label("Composite score")
        fig = bar_chart(
            ranked, x="display_name", y="composite_score",
            color_map={row["display_name"]: global_color_map[row["display_name"]] for _, row in ranked.iterrows()},
            sort_asc=False, height=380, y_label="Composite score (0–1)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        section_label("Per-metric breakdown")
        sel_m = st.selectbox(
            "Metric",
            avail_metrics,
            format_func=metric_label,
            key="leaderboard_metric_select",
            label_visibility="collapsed" # Hide label to align better with the other chart
        )
        
        plot_df = agg_df.dropna(subset=[sel_m]).sort_values(sel_m, ascending=(metric_direction(sel_m) == "lower"))
        fig2 = bar_chart(
            plot_df,
            x="display_name", y=sel_m,
            color_map={row["display_name"]: global_color_map[row["display_name"]] for _, row in plot_df.iterrows()},
            height=380,
            y_label=f"{metric_label(sel_m)} ({direction_arrow(sel_m)})",
        )
        st.plotly_chart(fig2, use_container_width=True)
