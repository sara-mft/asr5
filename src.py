# ─── TAB 1: Overview ──────────────────────────────────────────────────────────

def tab_overview(s_df: pd.DataFrame, d_df: pd.DataFrame, filters: dict) -> None:
    if s_df.empty:
        st.warning("No data matches the current filters.")
        return

    sel_groups = filters.get("sel_groups", ALL_GROUPS)
    
    # Force core metrics to ensure both scatter charts work perfectly
    base_keys = [m.key for g in sel_groups for m in METRICS_BY_GROUP.get(g, [])]
    metric_keys = list(set(base_keys + ["wer", "latency_sec", "ttfw_sec", "cost_usd"]))
    metric_keys = [k for k in metric_keys if k in s_df.columns]

    # Aggregate across datasets (mean)
    agg_df = _agg_summary(s_df, metric_keys, agg_fn="mean")

    # Filter out models by their Streaming tag
    is_stream = agg_df["categories"].apply(lambda x: "Streaming" in x if isinstance(x, list) else "Streaming" in str(x))
    df_stream = agg_df[is_stream]
    df_non = agg_df[~is_stream]

    has_wer = "wer" in agg_df.columns

    # ── Accuracy vs Speed Scatter Plots ──
    c_left, c_right = st.columns(2)
    
    with c_left:
        section_label("Non-Streaming Models (WER vs. Latency)")
        if has_wer and "latency_sec" in df_non.columns and not df_non.empty:
            plot_df = df_non.dropna(subset=["wer", "latency_sec"]).copy()
            if not plot_df.empty:
                has_cost = "cost_usd" in plot_df.columns and not plot_df["cost_usd"].isna().all()
                if has_cost:
                    plot_df["_bubble_size"] = plot_df["cost_usd"].fillna(0).clip(lower=0.0001)
                
                fig1 = px.scatter(
                    plot_df, x="latency_sec", y="wer", color="source",
                    size="_bubble_size" if has_cost else None,
                    hover_name="display_name", text="display_name",
                    labels={"latency_sec": "Avg. Latency (seconds)", "wer": "Avg. WER", "source": "Source"},
                    size_max=30 if has_cost else 10
                )
                fig1.update_traces(
                    textposition="top center", 
                    textfont=dict(size=10, color=PALETTE["text_muted"]), 
                    marker=dict(opacity=0.8, line=dict(width=1, color=PALETTE["bg_page"]))
                )
                fig1 = _chart_layout(fig1, height=420)
                # Ideal corner annotation
                fig1.add_annotation(
                    x=plot_df["latency_sec"].min() * 0.9, y=plot_df["wer"].min() * 0.9,
                    text="↙ Fast & Accurate", showarrow=False, font=dict(size=10, color=PALETTE["accent3"])
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Insufficient data to plot non-streaming models.")
        else:
            st.info("No non-streaming models selected, or missing WER/Latency data.")

    with c_right:
        section_label("Streaming Models (WER vs. TTFW)")
        if has_wer and "ttfw_sec" in df_stream.columns and not df_stream.empty:
            plot_df2 = df_stream.dropna(subset=["wer", "ttfw_sec"]).copy()
            if not plot_df2.empty:
                has_cost = "cost_usd" in plot_df2.columns and not plot_df2["cost_usd"].isna().all()
                if has_cost:
                    plot_df2["_bubble_size"] = plot_df2["cost_usd"].fillna(0).clip(lower=0.0001)
                
                fig2 = px.scatter(
                    plot_df2, x="ttfw_sec", y="wer", color="source",
                    size="_bubble_size" if has_cost else None,
                    hover_name="display_name", text="display_name",
                    labels={"ttfw_sec": "Avg. Time to First Word (sec)", "wer": "Avg. WER", "source": "Source"},
                    size_max=30 if has_cost else 10
                )
                fig2.update_traces(
                    textposition="top center", 
                    textfont=dict(size=10, color=PALETTE["text_muted"]), 
                    marker=dict(opacity=0.8, line=dict(width=1, color=PALETTE["bg_page"]))
                )
                fig2 = _chart_layout(fig2, height=420)
                # Ideal corner annotation
                fig2.add_annotation(
                    x=plot_df2["ttfw_sec"].min() * 0.9, y=plot_df2["wer"].min() * 0.9,
                    text="↙ Fast & Accurate", showarrow=False, font=dict(size=10, color=PALETTE["accent3"])
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Insufficient data to plot streaming models.")
        else:
            st.info("No streaming models selected, or missing TTFW data.")

    # ── Per-dataset WER bar ──
    if s_df["dataset"].nunique() > 1 and has_wer:
        section_label("WER by model × dataset")
        pivot = s_df.pivot_table(index="display_name", columns="dataset", values="wer", aggfunc="mean")
        
        fig3 = go.Figure()
        
        for dataset_col in pivot.columns:
            fig3.add_trace(go.Bar(
                name=dataset_col, x=pivot.index, y=pivot[dataset_col],
                text=[format_value("wer", v) for v in pivot[dataset_col]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            
        fig3.update_layout(barmode="group")
        fig3 = _chart_layout(fig3, height=450)
        st.plotly_chart(fig3, use_container_width=True)
