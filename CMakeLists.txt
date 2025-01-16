project(HistoToCTRegistrationPlugin)
cmake_minimum_required(VERSION 3.2.0)
message(STATUS "> Configuring ${PROJECT_NAME}")

#find_package(ImFusionLib COMPONENTS ImFusionML ImFusionReg ImFusionHistology ImFusionVision REQUIRED)

set(Sources
		HistoToCTRegistrationAlgorithm.cpp
		HistoToCTRegistrationPlugin.cpp
		)

set(Headers
		Common.h
		HistoToCTRegistrationAlgorithm.h
		HistoToCTRegistrationPlugin.h
		)

add_library(HistoToCTRegistrationPlugin SHARED ${Sources} ${UiSources} ${Headers})
target_link_libraries(HistoToCTRegistrationPlugin PRIVATE ImFusionLib ImFusionReg ImFusionML ImFusionVision ImFusionHistology)


#imfusion_set_common_target_properties()
#imfusion_provide_ide_instructions()

# install models as resources
imfusion_compile_resource_repository(HistoToCTRegistrationPluginResources
		FILES
		${CMAKE_CURRENT_SOURCE_DIR}/model/model.pt
		${CMAKE_CURRENT_SOURCE_DIR}/model/model-Torch.yaml
		BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model
		COMPRESS
		)

imfusion_set_common_target_properties(Plugin
	DEFINE_SYMBOL "IMFUSION_HISTOCTREG_DLL"
)

###########################################################
# Define install targets
###########################################################
imfusion_common_install(Plugin)
imfusion_sdk_register(Headers "IMFUSION_HISTOCTREG_API" "${CMAKE_CURRENT_SOURCE_DIR}/include")
imfusion_register_sdk_header_prefix("IMFUSION_HISTOCTREG_API")
define_source_groups_from_subdir(Sources "${CMAKE_CURRENT_SOURCE_DIR}/src" "src" "Source Files")
define_source_groups_from_subdir(Headers "${CMAKE_CURRENT_SOURCE_DIR}/include/HistoToCTRegistration" "include/HistoToCTRegistration" "Header Files")