import re
import logging
from thefuzz import process

class HeuristicParser:
    """
    Parses natural language queries using a series of heuristic rules to generate
    a query configuration, bypassing the need for an LLM for simple queries.
    """
    _TYPE_ALIASES = {
        "rgb": "did:rgb-image",
        "rgb image": "did:rgb-image",
        "merged lidar": "did:lidar-pointcloud-merged",
        "merged lidar point cloud": "did:lidar-pointcloud-merged",
        "lidar point cloud": "did:lidar-pointcloud-frame",
        "point cloud": "did:lidar-pointcloud-frame",
        "point clouds": None,  # expands to both merged + frame
        "lidar range panorama": "did:lidar-range-pano",
        "lidar reflectance panorama": "did:lidar-reflectance-pano",
        "lidar signal panorama": "did:lidar-signal-pano",
        "lidar nearir panorama": "did:lidar-nearir-pano",
        "ir false color": "did:ir-false-color-image",
        "ir temperature array": "did:ir-temperature-array",
        "ir counts": "did:ir-count-image",
        "temperature": "did:celsius-temperature",
        "relative humidity": "did:relative-humidity",
    }

    _POINTCLOUD_BOTH = ["did:lidar-pointcloud-merged", "did:lidar-pointcloud-frame"]

    _FILTER_ALIASES = {
        "property type": "property-type",
        "built form": "built-form",
        "accommodation type": "accommodation-type",
        "tenure": "tenure",
        "epc rating": "epc-rating",
        "epc": "epc-rating",
        "potential epc rating": "potential-epc-rating",
        "potential epc": "potential-epc-rating",
        "construction age band": "construction-age-band",
        "year built": "construction-age-band",
        "year built band": "construction-age-band",
        "age band": "construction-age-band",
        "was built": "construction-age-band",
        "built in": "construction-age-band",
        "building use": "building-use",
        "buildings used": "building-use",
        "main heating system": "main-heating-system",
        "heating": "main-heating-system",
        "main fuel type": "main-fuel-type",
        "fuel": "main-fuel-type",
        "energy source": "main-fuel-type",
        "wall type": "wall-type",
        "wall": "wall-type",
        "walls": "wall-type",
        "roof type": "roof-type",
        "roof": "roof-type",
        "roofs": "roof-type",
        "glazing type": "glazing-type",
        "glazing": "glazing-type",
        "loac group": "loac-group",
        "loac supergroup": "loac-supergroup",
        "listed building grade": "listed-building-grade",
        "listed building": "listed-building-grade",
        "heat risk quintile": "heat-risk-quintile",
        "imd19 income decile": "imd19-income-decile",
        "imd19 national decile": "imd19-national-decile",
        "wall insulation": "wall-insulation",  
        "roof insulation": "roof-insulation",
        "in conservation area": "in-conservation-area",
        "conservation area": "in-conservation-area",
        "conservation areas": "in-conservation-area",
        "epc score": "epc-score",
        "potential epc score": "potential-epc-score",
        "floor count": "floor-count",
        "number of floors": "floor-count",
        "floors": "floor-count",
        "basement floor count": "basement-floor-count",
        "number of basements": "basement-floor-count",
        "number of basement floors": "basement-floor-count",
        "number of habitable rooms": "number-of-habitable-rooms",
        "habitable rooms": "number-of-habitable-rooms",
        "easting": "easting",
        "x coordinate": "easting",
        "northing": "northing",
        "y coordinate": "northing",
        "total floor area": "total-floor-area",
        "floor area": "total-floor-area",
        "energy consumption": "energy-consumption",
        "energy use": "energy-consumption",
        "solar pv area": "solar-pv-area",
        "area for solar": "solar-pv-area",
        "solar pv potential": "solar-pv-potential",
        "solar potential": "solar-pv-potential",
        "average roof tilt": "average-roof-tilt",
        "roof tilt": "average-roof-tilt",
        "roof angle": "average-roof-tilt",
        "fuel poverty": "fuel-poverty"
    }

    _ENUM_FILTERS = {
        "property-type": {"flat", "house", "park-home-caravan"},
        "built-form": {"detached", "semi-detached", "end-terrace", "mid-terrace"},
        "accommodation-type": {"flat", "semi-detached-house", "detached-house", "end-terraced-house", "mid-terraced-house", "park-home-caravan"},
        "tenure": {"owner-occupied", "social-housing", "privately-rented"},
        "epc-rating": {"AB", "C", "D", "E", "FG"},
        "potential-epc-rating": {"AB", "C", "D", "E", "FG"},
        "construction-age-band": {"pre-1900", "1900-1929", "1930-1949", "1950-1966", "1967-1982", "1983-1995", "1996-2011", "2012-onwards"},
        "building-use": {"residential-only", "mixed-use"},
        "main-heating-system": {"boiler", "room-storage-heaters", "heat-pump", "communal", "none", "other"},
        "main-fuel-type": {"mains-gas", "electricity", "no-heating-system", "other"},
        "wall-type": {"cavity", "solid", "other"},
        "roof-type": {"pitched", "flat", "room-in-roof", "another-dwelling-above"},
        "glazing-type": {"single-partial", "secondary", "double-triple"},
        "loac-supergroup": {"A", "B", "C", "D", "E", "F", "G"},
        "loac-group": {"A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1", "D2", "D3", "E1", "E2", "F1", "F2", "G1", "G2"},
        "listed-building-grade": {"I", "II", "IIStar", "Unknown"},
    }

    _NUMERIC_FILTERS = [
        "heat-risk-quintile",
        "imd19-income-decile",
        "imd19-national-decile",
        "epc-score",
        "potential-epc-score",
        "floor-count",
        "basement-floor-count",
        "number-of-habitable-rooms",
        "easting",
        "northing",
        "total-floor-area",
        "energy-consumption",
        "solar-pv-area",
        "solar-pv-potential",
        "average-roof-tilt",
        "fuel-poverty"
    ]

    _BOOLEAN_FILTERS = [
        "wall-insulation",
        "roof-insulation",
        "in-conservation-area"
    ]

    _IMPLIED_FILTERS = {
        "flat": ("property-type", "flat", ["roof", "roofs"]),
        "flats": ("property-type", "flat", ["roof", "roofs"]),
        "house": ("property-type", "house"),
        "houses": ("property-type", "house"),
        "detached": ("built-form", "detached"),
        "semi-detached": ("built-form", "semi-detached"),
        "owner-occupied": ("tenure", "owner-occupied"),
        "privately-rented": ("tenure", "privately-rented"),
        "rented": ("tenure", "privately-rented"),
        "social-housing": ("tenure", "social-housing"),
    }
    _RELABELLED_GEO = {
        "administrative-area": "london-borough",
        "ward": "ward"
    }        


    def __init__(self, gazetteer: dict):
        self.gazetteer = gazetteer
        # Initialize state attributes that will be reset for each parse
        self.nl_query = ""
        self.nl_lower = ""
        self.config = {}
        self.found_primary_constraint = False

    def _initialize_parse(self, nl_query: str):
        """Resets the state for a new query."""
        self.nl_query = nl_query
        self.nl_lower = f" {nl_query.lower()} "
        self.config = {}
        self.found_primary_constraint = False

    def _clean_number(self, num_str: str) -> str:
        return re.sub(r"[^\d.]", "", num_str)

    def _map_year_to_band(self, year: int) -> str | None:
        if year < 1900: return "pre-1900"
        if 1900 <= year <= 1929: return "1900-1929"
        if 1930 <= year <= 1949: return "1930-1949"
        if 1950 <= year <= 1966: return "1950-1966"
        if 1967 <= year <= 1982: return "1967-1982" 
        if 1983 <= year <= 1995: return "1983-1995" 
        if 1996 <= year <= 2011: return "1996-2011"
        if year >= 2012: return "2012-onwards"
        return None
    
    def _find_identifiers(self):
        """Finds UPRN, ODS, TOID constraints."""
        if "uprn" in self.nl_lower or "uprns" in self.nl_lower:
            uprns = re.findall(r"\b\d{6,}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if uprns or csv_files:
                logging.info("Heuristic match: Detected UPRN query.")
                self.config.setdefault("identifier", {})["uprn"] = uprns + csv_files
                self.found_primary_constraint = True  

        elif "ods" in self.nl_lower:
            ods_codes = re.findall(r"\b[a-zA-Z]\d{5}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if ods_codes or csv_files:
                logging.info("Heuristic match: Detected ODS query.")
                self.config.setdefault("identifier", {})["ods"] = ods_codes + csv_files
                self.found_primary_constraint = True  

        elif "toid" in self.nl_lower or "toids" in self.nl_lower:
            toids = re.findall(r"\b\d{10,}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if toids or csv_files:
                logging.info("Heuristic match: Detected TOID query.")
                self.config.setdefault("identifier", {})["toid"] = toids + csv_files
                self.found_primary_constraint = True

    def _find_geographies(self):
        """Finds geographical constraints using codes, gazetteer, and fuzzy matching."""
        if "output area" in self.nl_lower or "oa" in self.nl_lower:
            oa_codes = re.findall(r"\bE\d{8}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if oa_codes or csv_files:
                logging.info("Heuristic match: Detected OA query.")
                self.config.setdefault("geography", {})["output-area"] = oa_codes + csv_files
                self.found_primary_constraint = True

        elif "lower layer super output area" in self.nl_lower or "lsoa" in self.nl_lower:
            lsoa_codes = re.findall(r"\bE\d{9}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if lsoa_codes or csv_files:
                logging.info("Heuristic match: Detected LSOA query.")
                self.config.setdefault("geography", {})["lower-layer-super-output-area"] = lsoa_codes + csv_files
                self.found_primary_constraint = True

        elif "ward" in self.nl_lower or "wards" in self.nl_lower:
            ward_codes = re.findall(r"\bE\d{7}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if ward_codes or csv_files:
                logging.info("Heuristic match: Detected Ward query.")
                self.config.setdefault("geography", {})["ward"] = ward_codes + csv_files
                self.found_primary_constraint = True    

        elif "london borough" in self.nl_lower or "borough" in self.nl_lower:
            borough_codes = re.findall(r"\bE09\d{4}\b", self.nl_query)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if borough_codes or csv_files:
                logging.info("Heuristic match: Detected London Borough query.")
                self.config.setdefault("geography", {})["administrative-area"] = borough_codes + csv_files
                self.found_primary_constraint = True

        elif "postcode" in self.nl_lower or "postcodes" in self.nl_lower:
            postcodes = re.findall(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", self.nl_query, re.IGNORECASE)
            csv_files = re.findall(r"[\w/.-]+\.csv", self.nl_query)
            if postcodes or csv_files:
                logging.info("Heuristic match: Detected Postcode query.")
                self.config.setdefault("geography", {})["postcode"] = [pc.upper().replace(" ", "") for pc in postcodes] + csv_files
                self.found_primary_constraint = True

        # Match exact names
        for geo_type, names_map in self.gazetteer.items():
            for name in names_map.keys():
                if re.search(r'\b' + re.escape(name) + r'\b', self.nl_lower):
                    code = names_map[name]
                    # Use the relabelled key if it exists, otherwise use original
                    key = self._RELABELLED_GEO.get(geo_type, geo_type)
                    self.config.setdefault("geography", {}).setdefault(key, []).append(code)
                    self.found_primary_constraint = True 

        # Fuzzy match geography names if no exact matches found yet
        if not self.config.get('geography'):
            # Find phrases following prepositions like "in", "at", "for"
            match = re.search(r'\b(in|at|for|near)\s+([\w\s,]+)', self.nl_query, re.IGNORECASE)
            if match:
                candidate_phrase = match.group(2).strip()
                
                best_overall_match = None
                highest_score = 0
                best_geo_type = None

                # Search all geography types to find the best possible match for the phrase
                for geo_type, names_map in self.gazetteer.items():
                    # process.extractOne finds the best fuzzy match in a list of choices
                    best_match, score = process.extractOne(candidate_phrase, names_map.keys())
                    
                    # If this match is better than any we've seen before, store it
                    if score > highest_score:
                        highest_score = score
                        best_overall_match = best_match
                        best_geo_type = geo_type
                
                # Use a high cutoff score to avoid false positives
                if highest_score >= 85:
                    logging.info(f"Heuristic FUZZY match: '{candidate_phrase}' -> '{best_overall_match}' (Score: {highest_score})")
                    code = self.gazetteer[best_geo_type][best_overall_match]
                    key = self._RELABELLED_GEO.get(best_geo_type, best_geo_type)
                    self.config.setdefault('geography', {}).setdefault(key, []).append(code)
                    self.found_primary_constraint = True

    def _find_filters(self):
        """Finds all types of secondary filters."""
        for alias, data in self._IMPLIED_FILTERS.items():
            if f" {alias} " in self.nl_lower:
                official_key, canonical_value = data[0], data[1]
                exclusion_keywords = data[2] if len(data) > 2 else []
                if any(f" {ex} " in self.nl_lower for ex in exclusion_keywords):
                    logging.info(f"Heuristic skip: Ignoring implied filter \"{alias}\" due to exclusion keyword.")
                    continue
                if official_key not in self.config.get("filter", {}):
                    logging.info(f"Heuristic match: Implied filter \"{alias}\" -> {official_key}: {canonical_value}")
                    self.config.setdefault("filter", {})[official_key] = [canonical_value]
        
        for alias, official_key in self._FILTER_ALIASES.items():
            if f" {alias} " in self.nl_lower:
                if official_key in self._ENUM_FILTERS.keys():
                    valid_options = self._ENUM_FILTERS.get(official_key, [])
                    found_values = []
                    for option in valid_options:
                        # Replace space for multi-word options like "owner occupied"
                        if f" {option.replace("-", " ")} " in self.nl_lower:
                            found_values.append(option)
                    if found_values:
                        self.config.setdefault("filter", {})[official_key] = found_values
                if official_key in self._BOOLEAN_FILTERS:
                    # Check for negation words immediately before the phrase
                    negation_pattern = rf"\b(no|not|without)\s+{re.escape(alias)}"
                    if re.search(negation_pattern, self.nl_lower):
                        logging.info(f"Heuristic match: Boolean filter \"{alias}\" set to FALSE")
                        self.config.setdefault("filter", {})[official_key] = ["false"]
                    else:
                        logging.info(f"Heuristic match: Boolean filter \"{alias}\" set to TRUE")
                        self.config.setdefault("filter", {})[official_key] = ["true"]

        # Numeric Filters (e.g., "epc score between 50 and 80")
        numeric_filter_pattern = "|".join(self._NUMERIC_FILTERS) # Creates a regex pattern like (epc score|total floor area|...)
        
        # Pattern for "between X and Y" or "from X to Y"
        range_pattern = re.compile(rf"({numeric_filter_pattern})\s+(?:between|from)\s+([\d,.]+)\s+(?:and|to)\s+([\d,.]+)", re.IGNORECASE)
        # Pattern for "over X", "at least X", etc.
        over_pattern = re.compile(rf"({numeric_filter_pattern})\s+(?:over|at least|greater than|more than)\s+([\d,.]+)", re.IGNORECASE)
        # Pattern for "under X", "at most X", etc.
        under_pattern = re.compile(rf"({numeric_filter_pattern})\s+(?:under|at most|less than)\s+([\d,.]+)", re.IGNORECASE)

        for match in range_pattern.finditer(self.nl_lower):
            key = self._FILTER_ALIASES.get(match.group(1).strip(), match.group(1).strip())
            val1 = self._clean_number(match.group(2))
            val2 = self._clean_number(match.group(3))
            self.config.setdefault("filter", {})[key] = [val1, val2]

        for match in over_pattern.finditer(self.nl_lower):
            key = self._FILTER_ALIASES.get(match.group(1).strip(), match.group(1).strip())
            val = self._clean_number(match.group(2))
            self.config.setdefault("filter", {})[key] = [val, "999999999"] 

        for match in under_pattern.finditer(self.nl_lower):
            key = self._FILTER_ALIASES.get(match.group(1).strip(), match.group(1).strip())
            val = self._clean_number(match.group(2))
            self.config.setdefault("filter", {})[key] = ["0", val]

        # Date/Decade Filters for Construction Age

        # Pattern for "built before 1980" or "built prior to 1980"
        built_pattern = re.search(r"built (?:before|prior to|until|up to|up until) (\d{4})", self.nl_lower)
        if built_pattern:
            year = int(built_pattern.group(1))
            bands = {"pre-1900", "1900-1929", "1930-1949", "1950-1966", "1967-1982", "1983-1995", "1996-2011", "2012-onwards"}
            bands_to_include = []
            target_band = self._map_year_to_band(year)
            for band in bands:
                if band < target_band:
                    bands_to_include.append(band)
            if bands_to_include:
                self.config.setdefault("filter", {})["construction-age-band"] = bands_to_include

        # Pattern for "built after 1980" or "built since 1980"
        built_pattern = re.search(r"built (?:after|since|from) (\d{4})", self.nl_lower)
        if built_pattern:
            year = int(built_pattern.group(1))
            bands = {"pre-1900", "1900-1929", "1930-1949", "1950-1966", "1967-1982", "1983-1995", "1996-2011", "2012-onwards"}
            bands_to_include = []
            target_band = self._map_year_to_band(year)
            for band in bands:
                if band > target_band:
                    bands_to_include.append(band)
            if bands_to_include:
                self.config.setdefault("filter", {})["construction-age-band"] = bands_to_include

        # Pattern for "built in the 1970s", "built in the 80s", etc.
        decade_pattern = re.search(r"built in the (\d{2})s", self.nl_lower)
        if decade_pattern:
            decade = int(decade_pattern.group(1))
            if 70 <= decade < 80: band = "1967-1982"
            elif 80 <= decade < 90: band = "1983-1995"
            else: band = None # Can add more decades
            if band:
                self.config.setdefault("filter", {})["construction-age-band"] = [band]

    def _find_asset_types(self):
        """Finds asset type keywords."""
        found_types = []

        pattern = r"\b(" + "|".join(re.escape(term) for term in self._TYPE_ALIASES) + r")\b"
        matches = re.findall(pattern, self.nl_lower)
        found_types.extend(self._TYPE_ALIASES[m] for m in matches if self._TYPE_ALIASES[m])

        if "point clouds" in self.nl_lower:
            found_types.extend(self._POINTCLOUD_BOTH)

        if found_types:
            self.config["types"] = found_types
    
    def parse(self, nl_query: str) -> dict | None:
        """
        Method to parse a query.
        """
        self._initialize_parse(nl_query)
        
        # Run all finders. They will populate self.config internally.
        self._find_identifiers()
        self._find_geographies()
        self._find_filters()
        self._find_asset_types()
        
        # Final check: only return a config if a primary constraint was found
        if self.found_primary_constraint:
            logging.info("Heuristic match successful.")
            return self.config

        return None


