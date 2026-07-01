aggregate = st.selectbox(
            "Aggregate across datasets by",
            ["mean", "median", "best-run", "worst-run"],
        )
        
        # ✅ FIX: Safely adapt slider bounds to the filtered model count
        n_models = s_df["model_key"].nunique()
        top_n = st.slider(
            "Top N models", 
            min_value=1, 
            max_value=n_models, 
            value=min(10, n_models)
        )
