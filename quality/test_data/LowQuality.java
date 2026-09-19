package com.example.benchmark;

import java.util.*;
import java.io.*;
import java.net.*;
import java.text.SimpleDateFormat;

public class DataProcessor {

    public List<String> data;
    public Map<String, Object> config;
    public Map<String, List<Double>> cache;
    public String dbUrl;
    public String apiKey;
    public int retryCount;
    public boolean debug;
    public List<String> log;
    public Map<String, Integer> counters;
    public double threshold;
    private PrintStream out;

    public DataProcessor() {
        data = new ArrayList<>();
        config = new HashMap<>();
        cache = new HashMap<>();
        dbUrl = "";
        apiKey = "";
        retryCount = 3;
        debug = false;
        log = new ArrayList<>();
        counters = new HashMap<>();
        threshold = 0.5;
        out = System.out;
    }

    public void loadData(String path) throws Exception {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(path));
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().length() > 0) {
                    if (!line.startsWith("#")) {
                        data.add(line.trim());
                        if (debug) {
                            log.add("loaded: " + line.trim());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.add("ERROR: " + e.getMessage());
            throw e;
        } finally {
            if (br != null) br.close();
        }
    }

    public Map<String, Object> processAll(int mode) {
        Map<String, Object> result = new HashMap<>();
        List<Double> vals = new ArrayList<>();
        List<String> errs = new ArrayList<>();
        int ok = 0;
        int fail = 0;

        for (int i = 0; i < data.size(); i++) {
            try {
                String[] parts = data.get(i).split(",");
                if (parts.length >= 2) {
                    double v = Double.parseDouble(parts[1]);
                    if (mode == 1) {
                        v = v * 2;
                        if (v > threshold) {
                            if (cache.containsKey(parts[0])) {
                                cache.get(parts[0]).add(v);
                            } else {
                                List<Double> l = new ArrayList<>();
                                l.add(v);
                                cache.put(parts[0], l);
                            }
                            vals.add(v);
                            ok++;
                        } else {
                            if (debug) log.add("below threshold: " + parts[0]);
                            fail++;
                        }
                    } else if (mode == 2) {
                        v = Math.sqrt(Math.abs(v));
                        vals.add(v);
                        ok++;
                    } else if (mode == 3) {
                        if (v > 0) {
                            v = Math.log(v);
                            vals.add(v);
                            ok++;
                        } else {
                            errs.add("negative value for log: " + parts[0]);
                            fail++;
                        }
                    } else {
                        vals.add(v);
                        ok++;
                    }
                } else {
                    errs.add("bad format: " + data.get(i));
                    fail++;
                }
            } catch (NumberFormatException e) {
                errs.add("parse error: " + data.get(i));
                fail++;
            }
        }

        result.put("values", vals);
        result.put("errors", errs);
        result.put("ok", ok);
        result.put("fail", fail);

        if (counters.containsKey("total_ok")) {
            counters.put("total_ok", counters.get("total_ok") + ok);
        } else {
            counters.put("total_ok", ok);
        }
        if (counters.containsKey("total_fail")) {
            counters.put("total_fail", counters.get("total_fail") + fail);
        } else {
            counters.put("total_fail", fail);
        }

        return result;
    }

    public double computeStats(List<Double> vals, String type) {
        if (vals == null || vals.isEmpty()) return 0.0;
        if (type.equals("mean")) {
            double s = 0;
            for (double v : vals) s += v;
            return s / vals.size();
        } else if (type.equals("median")) {
            List<Double> sorted = new ArrayList<>(vals);
            Collections.sort(sorted);
            int mid = sorted.size() / 2;
            if (sorted.size() % 2 == 0) {
                return (sorted.get(mid - 1) + sorted.get(mid)) / 2.0;
            } else {
                return sorted.get(mid);
            }
        } else if (type.equals("std")) {
            double mean = computeStats(vals, "mean");
            double ss = 0;
            for (double v : vals) ss += (v - mean) * (v - mean);
            return Math.sqrt(ss / vals.size());
        } else if (type.equals("max")) {
            double m = Double.MIN_VALUE;
            for (double v : vals) if (v > m) m = v;
            return m;
        } else if (type.equals("min")) {
            double m = Double.MAX_VALUE;
            for (double v : vals) if (v < m) m = v;
            return m;
        }
        return 0.0;
    }

    public String generateReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Report ===\n");
        sb.append("Date: " + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "\n");
        sb.append("Records: " + data.size() + "\n");
        sb.append("Cache entries: " + cache.size() + "\n");

        for (Map.Entry<String, List<Double>> entry : cache.entrySet()) {
            sb.append("\n" + entry.getKey() + ":\n");
            sb.append("  count: " + entry.getValue().size() + "\n");
            sb.append("  mean:  " + computeStats(entry.getValue(), "mean") + "\n");
            sb.append("  std:   " + computeStats(entry.getValue(), "std") + "\n");
            sb.append("  min:   " + computeStats(entry.getValue(), "min") + "\n");
            sb.append("  max:   " + computeStats(entry.getValue(), "max") + "\n");
        }

        if (counters.containsKey("total_ok")) {
            sb.append("\nTotal OK: " + counters.get("total_ok") + "\n");
        }
        if (counters.containsKey("total_fail")) {
            sb.append("Total Fail: " + counters.get("total_fail") + "\n");
        }
        sb.append("Log entries: " + log.size() + "\n");
        return sb.toString();
    }

    public void exportCsv(String path) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        pw.println("key,count,mean,std,min,max");
        for (Map.Entry<String, List<Double>> e : cache.entrySet()) {
            pw.print(e.getKey() + ",");
            pw.print(e.getValue().size() + ",");
            pw.print(computeStats(e.getValue(), "mean") + ",");
            pw.print(computeStats(e.getValue(), "std") + ",");
            pw.print(computeStats(e.getValue(), "min") + ",");
            pw.println(computeStats(e.getValue(), "max"));
        }
        pw.close();
    }

    public boolean sendToApi(String endpoint, String payload) {
        for (int attempt = 0; attempt < retryCount; attempt++) {
            try {
                URL url = new URL(endpoint);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Authorization", "Bearer " + apiKey);
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setDoOutput(true);
                OutputStream os = conn.getOutputStream();
                os.write(payload.getBytes());
                os.flush();
                os.close();
                int code = conn.getResponseCode();
                if (code == 200) {
                    log.add("API success: " + endpoint);
                    return true;
                } else {
                    log.add("API error " + code + " attempt " + (attempt + 1));
                }
            } catch (Exception e) {
                log.add("API exception: " + e.getMessage() + " attempt " + (attempt + 1));
            }
        }
        return false;
    }

    public void clearAll() {
        data.clear();
        cache.clear();
        log.clear();
        counters.clear();
    }
}
