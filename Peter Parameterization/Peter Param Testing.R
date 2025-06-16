library(tidyverse)
library(lubridate)
library(Metrics)
library(dplyr)
library(yardstick)
library(purrr)

setwd("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake")

# Load observed data
obs <- read_csv("Peter Inputs/observed_data2") %>%
  rename_with(tolower) %>%
  mutate(datetime = ymd_hms(datetime)) %>%
  drop_na(datetime, depth)

model_depths <- seq(0, 19, by = 0.5)
output_base <- "Peter Parameterization/outputs"
model_start <- ymd_hms("2024-06-10 09:00:00")

# Number of time steps 
n_time_steps <- nrow(read_csv(file.path(output_base, "Run_1/temp.csv"), col_names = FALSE))
time_grid <- model_start + hours(0:(n_time_steps - 1))

r2_results <- list()

run_folders <- list.dirs(output_base, full.names = TRUE, recursive = FALSE)
r_squared <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  return(1 - ss_res / ss_tot)
}
for (run_path in sort(run_folders)) {
  run_number <- as.integer(str_extract(basename(run_path), "\\d+"))
  cat("Processing Run", run_number, "\n")
  

  tryCatch({
    model_data <- list(
      wtemp = as.matrix(read_csv(file.path(run_path, "temp.csv"), col_names = FALSE)),
      do = as.matrix(read_csv(file.path(run_path, "do_mgL.csv"), col_names = FALSE)),
      doc = as.matrix(read_csv(file.path(run_path, "doc_mgL.csv"), col_names = FALSE)),
      secchi = as.matrix(read_csv(file.path(run_path, "secchi.csv"), col_names = FALSE))
    )
    
    r2_scores <- list(Run = run_number)
    
    for (var in names(model_data)) {
      obs_var <- obs %>% filter(!is.na(.data[[var]]))
      
      preds <- c()
      actuals <- c()
      
      for (i in seq_len(nrow(obs_var))) {
        row <- obs_var[i, ]
        obs_time <- row$datetime
        obs_depth <- row$depth
        
       
        if (!(obs_depth %in% model_depths)) next
        if (obs_time < min(time_grid) || obs_time > max(time_grid)) next
        
        time_before <- max(time_grid[time_grid <= obs_time])
        time_after <- min(time_grid[time_grid >= obs_time])
        
        i_before <- which(time_grid == time_before)
        i_after <- which(time_grid == time_after)
        d_index <- which(model_depths == obs_depth)
        
        val_before <- model_data[[var]][i_before, d_index]
        val_after <- model_data[[var]][i_after, d_index]
        
        if (time_before == time_after) {
          val_interp <- val_before
        } else {
          dt_total <- as.numeric(difftime(time_after, time_before, units = "secs"))
          dt_frac <- as.numeric(difftime(obs_time, time_before, units = "secs")) / dt_total
          val_interp <- val_before + dt_frac * (val_after - val_before)
        }
        
        preds <- c(preds, val_interp)
        actuals <- c(actuals, row[[var]])
      }
      
      if (length(preds) > 0) {
        r2_scores[[paste0("R2_", var)]] <- r_squared(actuals, preds) #rsq_trad(data.frame(obs = actuals, pred = preds))
        r2_scores[[paste0("n_", var)]] <- length(preds)
        rmse <- sqrt(mean((actuals - preds)^2, na.rm = TRUE))
        r2_scores[[paste0("RMSE_", var)]] <- rmse
      } else {
        r2_scores[[paste0("R2_", var)]] <- NA
        r2_scores[[paste0("n_", var)]] <- 0
        r2_scores[[paste0("RMSE_", var)]] <- NA
      }
    }
    
    valid_r2s <- unlist(r2_scores[grepl("^R2_", names(r2_scores))])
    r2_scores$Average_R2 <- if (length(valid_r2s[!is.na(valid_r2s)]) > 0) {
      mean(valid_r2s, na.rm = TRUE)
    } else {
      NA
    }
    
    r2_results[[length(r2_results) + 1]] <- r2_scores
    
  }, error = function(e) {
    cat("Run", run_number, "failed:", e$message, "\n")
  })
}

r2_df <- bind_rows(r2_results)
print(r2_df)

#analyzing 
top_models <- bind_rows(r2_results) %>%
  arrange(desc(Average_R2))
head(top_models, 20) 

top_models <- bind_rows(r2_results) %>%
  mutate(
    Average_R2_excl_doc = pmap_dbl(
      select(., R2_wtemp, R2_do, R2_secchi),
      ~ mean(c(...), na.rm = TRUE)
    )
  ) %>%
  arrange(desc(Average_R2_excl_doc))
print(top_models)

# View with observation counts
top_models %>%
  select(Run, starts_with("R2_"), starts_with("n_")) %>%
  head(20)

highest_r2_with_run <- r2_df %>%
  select(Run, starts_with("R2_")) %>%  # Select the Run column and R2 columns
  pivot_longer(cols = starts_with("R2_"), names_to = "Variable", values_to = "R2") %>%  # Reshape to long format
  group_by(Variable) %>%
  filter(R2 == max(R2, na.rm = TRUE)) %>%  # Get the row with the highest R2 for each variable
  ungroup()

# Print the highest R² values and corresponding run numbers
print(highest_r2_with_run)


####test 1 length####
run_folder <- "Peter Parameterization/outputs/Run_1"
obs <- read_csv("Peter Inputs/observed_data2") |> 
  mutate(datetime = ymd_hms(datetime)) |> 
  rename_with(tolower)

zmax <- 19
nx <- 2 * zmax
model_depths <- round(seq(0, zmax, length.out = nx), 2)

model_start <- ymd_hms("2024-06-10 00:09:00")
n_time_steps <- nrow(read_csv(file.path(run_folder, "temp.csv"), col_names = FALSE))
time_grid <- seq(model_start, by = "1 hour", length.out = n_time_steps)

# --- LOAD MODEL DATA ---
model_data <- list(
  wtemp = as.matrix(read_csv(file.path(run_folder, "temp.csv"), col_names = FALSE)),
  do = as.matrix(read_csv(file.path(run_folder, "do_mgL.csv"), col_names = FALSE)),
  doc = as.matrix(read_csv(file.path(run_folder, "doc_mgL.csv"), col_names = FALSE)),
  secchi = as.matrix(read_csv(file.path(run_folder, "secchi.csv"), col_names = FALSE))
)

# --- R² FUNCTION ---
r_squared <- function(obs, pred) {
  ss_res <- sum((obs - pred)^2)
  ss_tot <- sum((obs - mean(obs))^2)
  1 - ss_res / ss_tot
}

# --- PERFORM EVALUATION ---
results <- list()
for (var in c("wtemp", "do", "doc", "secchi")) {
  obs_var <- obs[!is.na(obs[[var]]), ]
  preds <- numeric()
  actuals <- numeric()
  
  for (i in seq_len(nrow(obs_var))) {
    row <- obs_var[i, ]
    obs_time <- row$datetime
    obs_depth <- row$depth
    
    if (!(obs_depth %in% model_depths)) next
    if (obs_time < min(time_grid) || obs_time > max(time_grid)) next
    
    time_before <- max(time_grid[time_grid <= obs_time])
    time_after <- min(time_grid[time_grid >= obs_time])
    
    i_before <- which(time_grid == time_before)
    i_after <- which(time_grid == time_after)
    d_index <- which(model_depths == obs_depth)
    
    val_before <- model_data[[var]][i_before, d_index]
    val_after <- model_data[[var]][i_after, d_index]
    
    if (time_before == time_after) {
      val_interp <- val_before
    } else {
      dt_total <- as.numeric(difftime(time_after, time_before, units = "secs"))
      dt_frac <- as.numeric(difftime(obs_time, time_before, units = "secs")) / dt_total
      val_interp <- val_before + dt_frac * (val_after - val_before)
    }
    
    preds <- c(preds, val_interp)
    actuals <- c(actuals, row[[var]])
  }
  
  if (length(preds) > 0) {
    r2 <- r_squared(actuals, preds)
    rmse <- sqrt(mean((actuals - preds)^2))
    results[[var]] <- list(R2 = r2, RMSE = rmse, n = length(preds))
  } else {
    results[[var]] <- list(R2 = NA, RMSE = NA, n = 0)
  }
}

results_df <- bind_rows(lapply(results, as.data.frame), .id = "Variable")
print(results_df)

####graphing####

run_nums <- c(1, 2, 3)
select_depths <- c(2, 8)  
vars_to_plot <- c("wtemp", "doc", "do", "secchi")

obs <- read_csv("Peter Inputs/observed_data2") %>%
  rename_with(tolower) %>%
  mutate(datetime = ymd_hms(datetime)) %>%
  drop_na(datetime, depth)

model_depths <- seq(0, 19, by = 0.5)
output_base <- "Peter Parameterization/outputs"
model_start <- ymd_hms("2024-06-10 09:00:00")

# Number of time steps 
n_time_steps <- nrow(read_csv(file.path(output_base, "Run_1/temp.csv"), col_names = FALSE))
time_grid <- model_start + hours(0:(n_time_steps - 1))

run_folders <- list.dirs(output_base, full.names = TRUE, recursive = FALSE)

for (run_path in sort(run_folders)) {
  run_number <- as.integer(str_extract(basename(run_path), "\\d+"))
  cat("Processing Run", run_number, "\n")
  
  # Load model output
  tryCatch({
    model_data <- list(
      wtemp = as.matrix(read_csv(file.path(run_path, "temp.csv"), col_names = FALSE)),
      do = as.matrix(read_csv(file.path(run_path, "do_mgL.csv"), col_names = FALSE)),
      doc = as.matrix(read_csv(file.path(run_path, "doc_mgL.csv"), col_names = FALSE)),
      secchi = as.matrix(read_csv(file.path(run_path, "secchi.csv"), col_names = FALSE))
    )
    

for (run_num in run_nums) {
  for (var in vars_to_plot) {
    
    model_file <- file.path(base_dir, as.character(run_num), paste0(var, ".csv"))
    
    if (!file.exists(model_file)) {
      cat("Missing:", model_file, "\n")
      next
    }
    
    # Load model data
    model_data <- read_csv(model_file) %>%
      mutate(datetime = ymd_hms(datetime)) %>%
      pivot_longer(-datetime, names_to = "depth", values_to = "value") %>%
      mutate(depth = as.numeric(gsub("X", "", depth))) %>%
      filter(depth %in% depths_to_plot)
    
    # Filter observed data
    obs_var <- obs %>%
      filter(!is.na(.data[[var]]), depth %in% depths_to_plot) %>%
      select(datetime, depth, value = all_of(var))
    
    # Plot
    p <- ggplot() +
      geom_line(data = model_data, aes(x = datetime, y = value, color = as.factor(depth)), size = 1) +
      geom_point(data = obs_var, aes(x = datetime, y = value, color = as.factor(depth)), shape = 21, fill = "white", size = 2) +
      labs(
        title = paste("Run", run_num, "-", var),
        x = "Date", y = var,
        color = "Depth (m)"
      ) +
      theme_minimal()
    
    print(p)
  }
}

#all copied
run_nums <- c(1, 2, 3)
select_depths <- c(2, 8)
vars_to_plot <- c("wtemp", "doc", "do", "secchi")
    
obs <- read_csv("Peter Inputs/observed_data2.csv") %>%
  rename_with(tolower) %>%
  mutate(datetime = ymd_hms(datetime)) %>%
  drop_na(datetime, depth)

model_depths <- seq(0, 19, by = 0.5)
    output_base <- "Peter Parameterization/outputs"
    model_start <- ymd_hms("2024-06-10 09:00:00")
    
    # Number of time steps to define time axis
    n_time_steps <- nrow(read_csv(file.path(output_base, "Run_1/temp.csv"), col_names = FALSE))
    time_grid <- model_start + hours(0:(n_time_steps - 1))
    
    # ----------------------------
    # Loop through runs and variables
    # ----------------------------
    
    for (run_num in run_nums) {
      run_path <- file.path(output_base, paste0("Run_", run_num))
      cat("Plotting Run", run_num, "\n")
      
      for (var in vars_to_plot) {
        model_file <- ""
        var_label <- var  # For plotting
        
        if (var == "wtemp") {
          model_file <- "temp.csv"
        } else if (var == "do") {
          model_file <- "do_mgL.csv"
        } else if (var == "doc") {
          model_file <- "doc_mgL.csv"
        } else if (var == "secchi") {
          model_file <- "secchi.csv"
        }
        
        full_path <- file.path(run_path, model_file)
        if (!file.exists(full_path)) {
          cat("Missing:", full_path, "\n")
          next
        }
        
        model_matrix <- as.matrix(read_csv(full_path, col_names = FALSE))
        
        # Check dimensions
        if (nrow(model_matrix) != length(time_grid)) {
          cat("Time mismatch in", var, "run", run_num, "\n")
          next
        }
        
        # Convert model matrix to tidy format
        model_df <- as_tibble(model_matrix)
        colnames(model_df) <- paste0("depth_", model_depths)
        model_df$datetime <- time_grid
        
        model_long <- model_df %>%
          pivot_longer(-datetime, names_to = "depth", values_to = "value") %>%
          mutate(depth = as.numeric(str_replace(depth, "depth_", ""))) %>%
          filter(depth %in% select_depths)
        
        # Get observed values for same variable and depths
        if (!var %in% colnames(obs)) {
          cat("Skipping", var, "– not in obs\n")
          next
        }
        
        obs_filtered <- obs %>%
          filter(depth %in% select_depths, !is.na(.data[[var]])) %>%
          select(datetime, depth, value = all_of(var))
        
        # Plot
        p <- ggplot() +
          geom_line(data = model_long, aes(x = datetime, y = value, color = factor(depth)), size = 1) +
          geom_point(data = obs_filtered, aes(x = datetime, y = value, color = factor(depth)), shape = 21, fill = "white", size = 2) +
          labs(
            title = paste("Run", run_num, "-", var_label),
            x = "Date", y = var_label,
            color = "Depth (m)"
          ) +
          theme_minimal()
        
        print(p)
      }
    }

 ####graphing 2####
