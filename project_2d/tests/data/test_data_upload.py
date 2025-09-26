import sys
from unittest.mock import patch


def test_data_upload_script_logic():
    with patch("azure.ai.ml.MLClient") as mock_mlclient_class, \
         patch("azure.identity.InteractiveBrowserCredential") as mock_credential_class, \
         patch("azure.ai.ml.entities.Data") as mock_data_class:
        # Setup mocks
        mock_mlclient = mock_mlclient_class.return_value
        mock_data_instance = mock_data_class.return_value

        # Import your data_upload.py AFTER patching, so code runs with mocks
        if "src.npecage.data.data_upload" in sys.modules:
            del sys.modules["src.npecage.data.data_upload"]  # reload if already imported
        import src.npecage.data.data_upload  # noqa: F401

        # Assert MLClient was created with InteractiveBrowserCredential instance
        mock_mlclient_class.assert_called_once()
        mock_credential_class.assert_called_once()

        # Assert Data() was created with expected args (you can add more precise checks if needed)
        mock_data_class.assert_called_once()

        # Assert create_or_update called once with the data instance
        mock_mlclient.data.create_or_update.assert_called_once_with(mock_data_instance)
