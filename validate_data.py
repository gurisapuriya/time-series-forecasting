# E:\proj\validate_data.py
import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

def validate_sales(values):
    df = pd.DataFrame({"value": values})
    context = gx.get_context()

    # Add pandas datasource
    datasource_name = "pandas_datasource"
    try:
        datasource = context.data_sources.add_pandas(name=datasource_name)
    except Exception:
        datasource = context.data_sources.get(name=datasource_name)

    # Add dataframe asset
    asset_name = "sales_data"
    data_asset = datasource.add_dataframe_asset(name=asset_name)

    # Create expectation suite
    suite_name = "sales_validation"
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # Add expectations
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeOfType(column="value", type_="float64")
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="value", min_value=0, strict_min=True)
    )
    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(min_value=4, max_value=1000)
    )

    # Create batch request with runtime data
    batch_request = data_asset.build_batch_request(options={"dataframe": df})

    # Set up validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )

    # Validate
    results = validator.validate()

    if results["success"]:
        return True, "Data valid."
    return False, results.to_json_dict()